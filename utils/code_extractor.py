import tempfile
from pathlib import Path

def extract_code_from_file(uploaded_file):
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()

    try:
        if filename.endswith('.pdf'):
            import pdfplumber
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            text = ""
            with pdfplumber.open(tmp_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    code_lines = [line for line in page_text.split('\n') 
                                  if line.strip() and any(k in line for k in ['{', '}', 'int ', 'void ', 'class ', '#include', '//', '/*'])]
                    text += '\n'.join(code_lines) + '\n'
            Path(tmp_path).unlink(missing_ok=True)
            return text.strip()

        elif filename.endswith(('.docx', '.doc')):
            from docx import Document
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            doc = Document(tmp_path)
            text = "\n".join([p.text for p in doc.paragraphs if any(k in p.text for k in ['{', '}', '#include', 'int ', 'class '])])
            Path(tmp_path).unlink(missing_ok=True)
            return text.strip()

        else:
            return content.decode("utf-8", errors="ignore").strip()
    except:
        return content.decode("utf-8", errors="ignore").strip()
