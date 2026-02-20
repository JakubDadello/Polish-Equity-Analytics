FROM public.ecr.aws/lambda/python:3.10

# --- Copy requirements file ---  
COPY requirements.txt .

# --- Install dependencies ---
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# --- Copy Lamba code --- 
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# --- Start the application ---- 
CMD ["lambda_function.lambda_handler"]