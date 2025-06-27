import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from dotenv import load_dotenv
from mistralai import Mistral
import pandas as pd
import uuid
import time
import re
from difflib import SequenceMatcher
import logging
import json

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Access the API key
API_KEY = os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    raise ValueError("MISTRAL_API_KEY is missing!")

# Paths & Directories
UPLOAD_DIR = "Uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatMistralAI(api_key=API_KEY, model="mistral-large-latest")

# Initialize Mistral client
mistral_client = Mistral(api_key=API_KEY)

# Initialize FastAPI
app = FastAPI()

# Store file data
file_data_store = {}

# Normalize contact numbers
def normalize_contact_number(number):
    return re.sub(r'[^0-9]', '', str(number))

# Fuzzy matching for names
def fuzzy_match(a, b, threshold=0.8):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

# Normalize field names
def normalize_field_name(field):
    return re.sub(r'[\[\]"\*•\-\s]+', ' ', field).strip().lower()

# Text Splitting Helper
def split_text(text, sheet_name, row_index):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_text(text)
    return [{"text": chunk, "sheet_name": sheet_name, "row_index": row_index} for chunk in chunks]

# Process Excel File
def process_excel_file(file_path):
    try:
        xl = pd.ExcelFile(file_path)
        all_text = []
        serial_data = {}

        for sheet_name in xl.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
            for index, row in df.iterrows():
                row_text = []
                row_dict = {}
                serial_numbers = []
                for col, value in row.items():
                    if pd.notna(value):
                        cleaned_value = str(value).strip()
                        row_dict[col] = cleaned_value
                        row_text.append(f"{col}: {cleaned_value}")
                        if col in ["Serial Number", "Print SR", "Sr. No."]:
                            serial_numbers.extend([s.strip() for s in cleaned_value.split(',') if s.strip()])
                if row_text:
                    combined_text = f"Sheet: {sheet_name}\nRow {index + 1}: {' | '.join(row_text)}"
                    all_text.extend(split_text(combined_text, sheet_name, index + 1))
                if serial_numbers:
                    for serial in serial_numbers:
                        if serial not in serial_data:
                            serial_data[serial] = []
                        serial_data[serial].append((sheet_name, index + 1, row_dict))
                else:
                    serial_data[f"row_{sheet_name}_{index + 1}"] = [(sheet_name, index + 1, row_dict)]
        return all_text, serial_data
    except Exception as e:
        raise Exception(f"Error processing Excel file: {str(e)}")

# Check Available Models
@app.get("/check_models/")
async def check_available_models():
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = mistral_client.models.list()
                models = [model.id for model in response.data]
                return {"message": "Successfully retrieved available models", "models": models}
            except Exception as e:
                if "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = 4 * (2 ** attempt)
                        time.sleep(wait_time)
                        continue
                raise e
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    except Exception as e:
        if "401" in str(e):
            raise HTTPException(status_code=401, detail="Invalid MISTRAL_API_KEY.")
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# List Uploaded Files
@app.get("/files/")
async def list_uploaded_files():
    return {"files": [{"file_id": file_id, "filename": data["filename"]} for file_id, data in file_data_store.items()]}

# Upload & Process Multiple Excel Files
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    global file_data_store
    uploaded_files = []
    allowed_extensions = [".xlsx"]

    for file in files:
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Unsupported file type for {file.filename}.")

        file_id = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        faiss_index_path = f"faiss_index_{file_id}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            chunks, serial_data = process_excel_file(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file {file.filename}: {str(e)}")

        if not chunks:
            raise HTTPException(status_code=400, detail=f"No text found in file {file.filename}.")

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"sheet_name": chunk["sheet_name"], "row_index": chunk["row_index"]} for chunk in chunks]
        vector_db = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_db.save_local(faiss_index_path)

        file_data_store[file_id] = {
            "vector_db": vector_db,
            "file_path": file_path,
            "all_rows_data": serial_data,
            "filename": file.filename
        }
        uploaded_files.append(file.filename)
        logger.info(f"Successfully processed file: {file.filename} with ID: {file_id}")

    return {"message": f"{len(uploaded_files)} file(s) uploaded!", "filenames": uploaded_files}

# Dynamic field categorization
def categorize_field(field):
    field = field.lower()
    categories = {
        "Branch Information": ["branch name", "branch code", "office"],
        "Hardware Details": ["hardware", "serial number", "qty", "quantity", "print sr", "sr. no."],
        "Purchase Order": ["po number", "purchase order"],
        "Location Details": ["address", "address 2", "city", "district", "state", "pincode", "location", "name of court complex", "court name"],
        "Contact Information": ["prefix", "name of contact person", "dsa/tsa", "contact number", "contact 1", "contact 2", "contact 3", "phone", "mobile"],
        "Delivery Details": ["delivery status", "delivery date", "courier", "lr / awb", "tracking id", "tracking number", "dispatch date", "shipping", "pod"],
        "Financial Information": ["rate", "price", "tax", "cgst", "sgst", "total gst", "inv total", "amount"],
        "Date Information": ["last date of delivery as per po", "extended date requested", "actual date of delivery", "dc date", "dispatch date", "hardcopy received date", "installation date"],
        "Hardcopy Information": ["hardcopy status", "hardcopy courier", "hardcopy tracking", "softcopy", "hardcopy received date", "hardcopy remarks", "remarks"],
        "Additional Information": ["dc number", "calling remark", "remark (call)", "kairee remarks"]
    }
    emoji_map = {
        "Branch Information": "🏢",
        "Hardware Details": "🖨️",
        "Purchase Order": "📄",
        "Location Details": "📍",
        "Contact Information": "📞",
        "Delivery Details": "🚚",
        "Financial Information": "💰",
        "Date Information": "📅",
        "Hardcopy Information": "📌",
        "Additional Information": "📌"
    }
    for category, keywords in categories.items():
        if any(keyword in field for keyword in keywords):
            return category, emoji_map[category]
    return "Other Details", "📋"

# Merge row data from multiple sheets
def merge_row_data(rows):
    merged = {}
    for sheet, row_index, row in rows:
        for key, value in row.items():
            if key not in merged or (value and value.lower() not in ["", "nan", "not specified"]):
                merged[key] = value
    return merged

# Format LLM response
def format_response(response_text, serial_number=None, contact_number=None, requested_fields=None, sheet_name=None, filename=None, context_row=None):
    if not response_text.strip() or "No matching data found" in response_text or "⚠️" in response_text:
        return None

    lines = response_text.split('\n')
    section_data = {}
    current_section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("###") or line.startswith("####"):
            if ":" in line or line.startswith("### Requested Field"):
                current_section = re.sub(r'^#+|\s*:\s*$', '', line).strip()
            continue
        if line.startswith("- ") or line.startswith("• "):
            clean_line = re.sub(r'^- |^• |\*\*', '', line).strip()
            if ": " in clean_line:
                key, value = clean_line.split(": ", 1)
                key = re.sub(r'\*\*', '', key).strip()
                value = value.strip()
                if value.lower() in ["", "nan", "not specified"] and context_row:
                    normalized_key = normalize_field_name(key)
                    if normalized_key in [normalize_field_name(k) for k in context_row]:
                        for k, v in context_row.items():
                            if normalize_field_name(k) == normalized_key and v:
                                value = v
                                break
                if value and value.lower() not in ["", "nan", "not specified"]:
                    section_data[key] = value

    formatted_response = []
    if serial_number:
        formatted_response.append(f"**🔍 Full Details for Serial Number: {serial_number} (File: {filename})**")
    elif contact_number:
        formatted_response.append(f"**📞 Full Details for Contact Number: {contact_number} (File: {filename})**")
    else:
        formatted_response.append(f"**🔍 Query Results from {sheet_name or 'All Sheets'} (File: {filename})**")
    formatted_response.append("")

    if requested_fields and requested_fields != ["all"]:
        found_fields = False
        formatted_response.append("**Requested Field**")
        for field in requested_fields:
            field_found = False
            normalized_field = normalize_field_name(field)
            for key, value in section_data.items():
                normalized_key = normalize_field_name(key)
                if normalized_field == normalized_key:
                    formatted_response.append(f"• {key}: {value}")
                    found_fields = True
                    field_found = True
                    break
            if not field_found and context_row:
                for k, v in context_row.items():
                    if normalize_field_name(k) == normalized_field and v:
                        formatted_response.append(f"• {k}: {v}")
                        found_fields = True
                        field_found = True
                        break
            if not field_found:
                formatted_response.append(f"• No matching data found for '{field}'")
        if not found_fields:
            return None
        formatted_response.append("")
        return "\n".join(formatted_response)

    categorized_data = {}
    for key, value in section_data.items():
        category, emoji = categorize_field(key)
        if category not in categorized_data:
            categorized_data[category] = []
        if key in ["Serial Number", "Print SR", "Sr. No."] and serial_number:
            value = serial_number
        elif key in ["Contact Number", "Contact 1", "Contact 2", "Contact 3"] and contact_number:
            value = contact_number
        elif key in ["Rate", "Price", "CGST", "SGST", "Total GST", "Inv Total"]:
            value = value.replace("₹", "").strip()
            value = f"₹{value}" if value else value
        elif key == "Tax":
            key = "Tax %"
        elif key == "LR / AWB":
            key = "Tracking ID"
        categorized_data[category].append(f"• {key}: {value}")

    for category, items in sorted(categorized_data.items()):
        if items:
            emoji = categorize_field(category)[1]
            formatted_response.append(f"**{emoji} {category}**")
            formatted_response.extend(sorted(items))
            formatted_response.append("")

    return "\n".join(formatted_response)

# Query Model
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_document(request: QueryRequest):
    global file_data_store

    if not file_data_store:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    intent_prompt = (
        f"You are an AI assistant analyzing a user query about Excel files with hardware data (scanners, printers). "
        f"The files contain columns like Print SR, Sr. No., Branch Name, Branch Code, DC Number, DC Date, Hardware, PO Number, "
        f"State, District, Name of Court Complex, Court Name, Address, Address 2, City, Pincode, QTY, Prefix, "
        f"Name of Contact Person, DSA/TSA, Contact 1, Contact 2, Contact 3, Contact Number, Serial Number, Qty, "
        f"Courier, LR / AWW, Tracking ID, Dispatch Date, Delivery Date, Delivery Status, POD, Installation Date, "
        f"Installation Status, SoftCopy, HardCopy Courier, HardCopy Tracking, HardCopy Received Date, HardCopy Status, "
        f"HardCopy Remarks, Remarks, and any custom columns. Handle synonyms and variations case-insensitively: "
        f"- 'address', 'location', 'address 2' for Address, Address 2\n"
        f"- 'phone', 'mobile', 'contact', 'contact number', 'contact 1' for Contact Number, Contact 1, Contact 2, Contact 3\n"
        f"- 'branch', 'office' for Branch Name, Branch Code\n"
        f"- 'courier', 'shipping' for Courier\n"
        f"- 'tracking id', 'tracking number', 'lr / awb' for LR / AWB, Tracking ID\n"
        f"- 'dispatch date', 'shipping date' for Dispatch Date (distinct from DC Date)\n"
        f"- 'delivery date', 'delivered' for Delivery Date (distinct from DC Date)\n"
        f"- 'dc', 'delivery challan' for DC Number, DC Date (distinct from Dispatch Date and Delivery Date)\n"
        f"- 'serial', 'sr. no.', 'print sr' for Serial Number, Sr. No., Print SR\n"
        f"- 'contact person', 'name of contact' for Name of Contact Person\n"
        f"- 'po', 'purchase order' for PO Number\n"
        f"Prioritize exact matches for serial numbers and contact numbers. For names, allow fuzzy matching. "
        f"Distinguish clearly between DC Date, Dispatch Date, and Delivery Date. "
        f"Determine:\n"
        f"1. Intent: 'full_details' for complete info, 'specific_fields' for certain columns, or 'summary' for aggregated data.\n"
        f"2. Key value: Extract the exact serial number, contact number, branch, contact person name, state, city, pincode, etc.\n"
        f"3. Key type: Identify as 'serial_number', 'contact_number', 'branch', 'state', 'city', 'pincode', 'contact_person', or 'none'.\n"
        f"4. Requested fields: List specific fields or 'all' for full details.\n"
        f"5. Sheet: Identify the sheet if mentioned or 'none' if not specified.\n"
        f"Respond in this format:\n"
        f"Intent: <full_details|specific_fields|summary>\n"
        f"Key value: <value or none>\n"
        f"Key type: <serial_number|contact_number|branch|state|city|pincode|contact_person|none>\n"
        f"Requested fields: <comma-separated list or 'all'>\n"
        f"Sheet: <sheet name or none>\n"
        f"Query: {request.query}"
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            intent_response = llm.invoke(intent_prompt)
            intent_result = intent_response.content if hasattr(intent_response, 'content') else str(intent_response)
            logger.debug(f"Intent detection result: {intent_result}")
            break
        except Exception as e:
            logger.error(f"Error in intent detection, attempt {attempt + 1}: {str(e)}")
            if "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 4 * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                raise HTTPException(status_code=429, detail="Rate limit exceeded.")
            raise HTTPException(status_code=500, detail=f"Error analyzing query intent: {str(e)}")

    intent = "specific_fields"
    key_value = "none"
    key_type = "none"
    requested_fields = []
    sheet_name = "none"

    for line in intent_result.split('\n'):
        line = line.strip()
        if line.startswith("Intent:"):
            intent = line.split(": ", 1)[1].strip()
        elif line.startswith("Key value:"):
            key_value = line.split(": ", 1)[1].strip()
        elif line.startswith("Key type:"):
            key_type = line.split(": ", 1)[1].strip()
        elif line.startswith("Requested fields:"):
            fields = line.split(": ", 1)[1].strip()
            if fields == "all":
                requested_fields = ["all"]
            else:
                try:
                    fields_list = json.loads(fields) if fields.startswith('[') else fields.split(',')
                    requested_fields = [f.strip() for f in fields_list if f.strip()]
                except json.JSONDecodeError:
                    requested_fields = [f.strip() for f in fields.split(',') if f.strip()]
        elif line.startswith("Sheet:"):
            sheet_name = line.split(": ", 1)[1].strip()
            if sheet_name == "none":
                sheet_name = None

    logger.debug(f"Parsed intent: {intent}, key_value: {key_value}, key_type: {key_type}, requested_fields: {requested_fields}, sheet: {sheet_name}")

    combined_response = []
    exact_match_found = False

    for file_id, file_data in file_data_store.items():
        vector_db = file_data["vector_db"]
        serial_data = file_data["all_rows_data"]
        filename = file_data["filename"]
        context_rows = []
        matched_serial = None
        matched_contact = None
        matched_rows = []

        if key_type == "serial_number" and key_value != "none":
            if key_value in serial_data:
                context_rows = serial_data[key_value]
                matched_serial = key_value
                exact_match_found = True
                logger.debug(f"Exact match found for serial number: {key_value} in file {filename}")
                matched_rows.extend(context_rows)

        if key_type == "contact_number" and key_value != "none":
            normalized_query = normalize_contact_number(key_value)
            for serial, rows in serial_data.items():
                for sheet, row_index, row in rows:
                    if any(col in ["Contact 1", "Contact 2", "Contact 3", "Contact Number"] for col in row):
                        for col in ["Contact 1", "Contact 2", "Contact 3", "Contact Number"]:
                            if col in row and row[col]:
                                cell_value = str(row[col]).strip()
                                cell_contacts = [normalize_contact_number(c.strip()) for c in cell_value.split(',')] if ',' in cell_value else [normalize_contact_number(cell_value)]
                                if normalized_query in cell_contacts:
                                    if (sheet, row_index, row) not in matched_rows:
                                        matched_rows.append((sheet, row_index, row))
                                        matched_contact = key_value
                                        logger.debug(f"Matched contact number: {key_value} in row {row_index}, sheet {sheet}, file {filename}")

        if not matched_rows and key_value != "none" and key_type != "serial_number":
            for serial, rows in serial_data.items():
                match_found = False
                for sheet, row_index, row in rows:
                    if key_type == "contact_person" and "Name of Contact Person" in row:
                        cell_value = str(row["Name of Contact Person"]).strip()
                        if fuzzy_match(key_value, cell_value):
                            matched_rows.append((sheet, row_index, row))
                            match_found = True
                            logger.debug(f"Matched contact person: {key_value} in row {row_index}, sheet {sheet}, file {filename}")
                    elif key_type in ["branch", "state", "city", "pincode"]:
                        target_columns = {
                            "branch": ["Branch Name", "Branch Code"],
                            "state": ["State"],
                            "city": ["City"],
                            "pincode": ["Pincode"]
                        }
                        for col in target_columns.get(key_type, []):
                            if col in row and row[col]:
                                cell_value = str(row[col]).strip()
                                if key_value.lower() in cell_value.lower():
                                    matched_rows.append((sheet, row_index, row))
                                    match_found = True
                                    logger.debug(f"Matched {key_type}: {key_value} in row {row_index}, sheet {sheet}, file {filename}")
                                    break
                    elif key_type == "none" and requested_fields and requested_fields != ["all"]:
                        for field in requested_fields:
                            for col in row:
                                if normalize_field_name(field) == normalize_field_name(col) and row[col]:
                                    cell_value = str(row[col]).strip()
                                    if key_value.lower() in cell_value.lower():
                                        matched_rows.append((sheet, row_index, row))
                                        match_found = True
                                        logger.debug(f"Matched field {field} with value {key_value} in row {row_index}, sheet {sheet}, file {filename}")
                                        break
                            if match_found:
                                break
                    if match_found and sheet_name and sheet.lower() != sheet_name.lower():
                        matched_rows = []
                        break
                    if match_found and intent != "summary":
                        break
                if match_found and intent != "summary":
                    break

        if not matched_rows:
            docs = vector_db.similarity_search_with_score(request.query, k=5)
            for doc, score in docs:
                if score < 0.5:
                    continue
                sheet = doc.metadata.get("sheet_name")
                row_index = doc.metadata.get("row_index")
                if sheet_name and sheet_name.lower() != sheet.lower():
                    continue
                for serial, rows in serial_data.items():
                    for s_name, r_index, row in rows:
                        if s_name == sheet and r_index == row_index:
                            if key_type == "serial_number" and any(col in row for col in ["Serial Number", "Print SR", "Sr. No."]):
                                match_found = False
                                for col in ["Serial Number", "Print SR", "Sr. No."]:
                                    if col in row:
                                        cell_value = str(row[col]).strip()
                                        cell_serials = [s.strip() for s in cell_value.split(',')] if ',' in cell_value else [cell_value]
                                        if key_value in cell_serials:
                                            match_found = True
                                            matched_serial = key_value
                                            break
                                if not match_found:
                                    continue
                            elif key_type == "contact_number" and any(col in row for col in ["Contact 1", "Contact 2", "Contact 3", "Contact Number"]):
                                normalized_query = normalize_contact_number(key_value)
                                contact_match = False
                                for col in ["Contact 1", "Contact 2", "Contact 3", "Contact Number"]:
                                    if col in row and row[col]:
                                        cell_value = str(row[col]).strip()
                                        cell_contacts = [normalize_contact_number(c.strip()) for c in cell_value.split(',')] if ',' in cell_value else [normalize_contact_number(cell_value)]
                                        if normalized_query in cell_contacts:
                                            contact_match = True
                                            matched_contact = key_value
                                            break
                                if not contact_match:
                                    continue
                            elif key_type == "contact_person" and "Name of Contact Person" in row:
                                cell_value = str(row["Name of Contact Person"]).strip()
                                if not fuzzy_match(key_value, cell_value):
                                    continue
                            elif key_type in ["branch", "state", "city", "pincode"]:
                                target_columns = {
                                    "branch": ["Branch Name", "Branch Code"],
                                    "state": ["State"],
                                    "city": ["City"],
                                    "pincode": ["Pincode"]
                                }
                                match_found = False
                                for col in target_columns.get(key_type, []):
                                    if col in row and row[col]:
                                        cell_value = str(row[col]).strip()
                                        if key_value.lower() in cell_value.lower():
                                            match_found = True
                                            break
                                if not match_found:
                                    continue
                            elif key_type == "none" and requested_fields and requested_fields != ["all"]:
                                match_found = False
                                for field in requested_fields:
                                    for col in row:
                                        if normalize_field_name(field) == normalize_field_name(col) and row[col]:
                                            cell_value = str(row[col]).strip()
                                            if key_value.lower() in cell_value.lower():
                                                match_found = True
                                                break
                                    if match_found:
                                        break
                                if not match_found:
                                    continue
                            if (s_name, r_index, row) not in matched_rows:
                                matched_rows.append((s_name, r_index, row))
                                logger.debug(f"FAISS matched row {row_index} in sheet {s_name}, file {filename}")
                            if intent != "summary":
                                break
                    if matched_rows and intent != "summary":
                        break

        if matched_rows:
            context_rows = matched_rows
            merged_row = merge_row_data(context_rows)
            context = "\n\n".join([f"Sheet: {sheet}\nRow {row_index}: {', '.join([f'{k}: {v}' for k, v in row.items()])}" for sheet, row_index, row in context_rows])
            logger.debug(f"Context for LLM (file {filename}): {context}")

            response_prompt = (
                f"You are an assistant providing precise answers based on an Excel file with hardware data. "
                f"The file '{filename}' contains columns like Print SR, Sr. No., Branch Name, Branch Code, DC Number, DC Date, Hardware, PO Number, "
                f"State, District, Name of Court Complex, Court Name, Address, Address 2, City, Pincode, QTY, Prefix, "
                f"Name of Contact Person, DSA/TSA, Contact 1, Contact 2, Contact 3, Contact Number, Serial Number, Qty, "
                f"Courier, LR / AWB, Tracking ID, Dispatch Date, Delivery Date, Delivery Status, POD, Installation Date, "
                f"Installation Status, SoftCopy, HardCopy Courier, HardCopy Tracking, HardCopy Received Date, HardCopy Status, "
                f"HardCopy Remarks, Remarks, and any custom columns. Follow these guidelines:\n"
                f"1. Format responses in markdown with sections and emojis, dynamically categorizing all columns.\n"
                f"2. Use bullet points (-) under each heading.\n"
                f"3. For specific fields, return only those under 'Requested Field'.\n"
                f"4. For 'full details', include all non-empty columns, grouped dynamically.\n"
                f"5. Match serial numbers exactly.\n"
                f"6. Skip empty, 'nan', or 'Not specified' values.\n"
                f"7. Distinguish DC Date, Dispatch Date, and Delivery Date.\n"
                f"8. Map 'LR / AWB' as 'Tracking ID'.\n"
                f"Query: {request.query}\n"
                f"Intent: {intent}\n"
                f"Key value: {key_value}\n"
                f"Key type: {key_type}\n"
                f"Requested fields: {', '.join(requested_fields) if requested_fields else 'none'}\n"
                f"Sheet: {sheet_name or 'none'}\n"
                f"Context: {context}"
            )

            for attempt in range(max_retries):
                try:
                    response = llm.invoke(response_prompt)
                    answer = response.content if hasattr(response, 'content') else str(response)
                    logger.debug(f"Raw LLM response (file {filename}): {answer}")
                    formatted_answer = format_response(answer, matched_serial, matched_contact, requested_fields, sheet_name, filename, merged_row)
                    if formatted_answer:
                        combined_response.append(formatted_answer)
                    break
                except Exception as e:
                    logger.error(f"Error in query processing for file {filename}, attempt {attempt + 1}: {str(e)}")
                    if "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = 4 * (2 ** attempt)
                            time.sleep(wait_time)
                            continue
                        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
                    raise HTTPException(status_code=500, detail=f"Error processing query for file {filename}: {str(e)}")

    if not combined_response:
        return {"answer": "No matching data found across all files", "token_count": 0}

    final_response = "\n\n---\n\n".join(combined_response)
    logger.debug(f"Final combined response: {final_response}")

    # Get token count
    token_count = len(final_response.split())

    return {"answer": final_response, "token_count": token_count}

# Clear Uploaded Files
@app.delete("/clear_files/")
async def clear_files():
    global file_data_store
    try:
        for file_id, file_data in file_data_store.items():
            file_path = file_data["file_path"]
            faiss_index_path = f"faiss_index_{file_id}"
            if os.path.exists(file_path):
                os.remove(file_path)
            if os.path.exists(faiss_index_path):
                shutil.rmtree(faiss_index_path)
        file_data_store = {}
        return {"message": "All files cleared successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing files: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
