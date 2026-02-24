from fastapi import APIRouter,UploadFile,File,HTTPException
from fastapi.responses import JSONResponse
import tempfile,os,uuid,shutil
from app.services.pdf_service import extract_text_with_coords

router=APIRouter(prefix="/api")
upload_registry={}
@router.post("/upload")
async def upload(file:UploadFile=File(...)):
    if file.content_type!="application/pdf":
        raise HTTPException(status_code=400,detail="Only PDF files are allowed")
    
    contents=await file.read()
    max_size=10*1024*1024
    if len(contents)>max_size:
        raise HTTPException(status_code=400,detail=f"File size exceeds 10MB limit"f"Your file: {len(contents)/(1024*1024):.1f}MB")
    
    upload_id=str(uuid.uuid4())
    
    temp_dir==tempfile.mkdtemp()
    temp_path=os.path.join(temp_dir,file.filename)
    
    try:
        with open(temp_path,"wb") as f:
            f.write(contents)
        
        extraction_result=extract_text_with_coords(temp_path)
        
        upload_registry[upload_id]={
            "file_path":temp_path,
            "filename":file.filename,
            "temp_dir":temp_dir,
        }
    
        return JSONResponse(content={"upload_id":upload_id,
                                     "filename":file.filename,
                                    "num_pages":extraction_result["num_pages"],
                                    "text_spans":extraction_result["spans"],
                                    "total_spans":len(extraction_result["spans"])})
    except Exception as e:
        shutil.rmtree(temp_dir,ignore_errors=True)
        raise HTTPException(status_code=500,detail=f"Error processing PDF: {str(e)}")
