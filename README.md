# RAG_QnA
# Demo Link ``
## Instructions to run the app locally.

### Clone the repo.
    git clone https://github.com/charucjoshi/RAG_QnA.git
### Create .env file in the cloned directory and add the following variables.
    OPENAI_API_KEY=your-openai-api-key
    TYPESENSE_API_KEY=your-typesense-api-key-with-correct-access-permissions
### Install requirements.
    pip install -r requirements.txt
### Ensure typesense server is running on port 8108.
### Run the application.
    chainlit run app.py

