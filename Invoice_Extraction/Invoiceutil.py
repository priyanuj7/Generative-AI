from langchain.chat_models import ChatOpenAI
from pypdf import PdfReader
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.llm import LLMChain
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0, max_tokens = 2000)


def create_docs(user_pdf_list):
    """
    This function is used to extract the invoice data from the given PDF Files. It uses
    the langchain agent to extract the data from the given PDF Files
    """

    df = pd.DataFrame(
        {
            'Invoice No': pd.Series(dtype = 'str'),
            'BILLED TO': pd.Series(dtype = 'str'),
            'Item': pd.Series(dtype = 'str'),
            'Quantity': pd.Series(dtype = 'str'),
            'Date': pd.Series(dtype = 'str'),
            'Unit Price': pd.Series(dtype = 'str'),
            'Phone no': pd.Series(dtype = 'str'),
            'Total': pd.Series(dtype = 'str'),
            'Delivery charges': pd.Series(dtype = 'str')
        }
    )

    for filename in user_pdf_list:
        
        texts = ""
        print("Processing ", filename)
        pdf_reader = PdfReader(filename)
        for page in pdf_reader.pages:
            texts+= page.extract_text()


        template = """
        Extract all the following values: invoice_no, billed_to, item, quantity, date, unit_price, customer_contact, total_price
        and delivery charges from the following invoice content:
        {texts}
        The fields and values in the above content may be jumbled as they are extracted from a PDF. Please use your judgement to align
        the fields and values correctly based on the fields asked for in the question mentioned above.
        Expected output format:
        {{'Invoice No': 'xxxxxx', 'BILLED TO': 'xxxxxxx', 'Item':'xxxxxxxx xxx xxxx', 'Quantity':'x', 'Date':'dd/dd/yyyy', 
        'Unit Price':'xxx', Phone no': '+xx-xxxxxxxxxx', 'Total': 'xxxx', 'Delivery charges': 'xx'}}
        Remove any dollar symbols or currency symbols from the extracted values
        Extract the value from the pdf and do not create random information.
        """

        prompt = PromptTemplate.from_template(template)

        llm = ChatOpenAI(temperature = 0, model_name = 'gpt-3.5-turbo')

        chain = LLMChain(llm = llm, prompt = prompt)

        data_dict = chain.run(texts)

        print("Dict:...", data_dict)

        new_row_df = pd.DataFrame([eval(data_dict)], columns = df.columns)
        df = pd.concat([df, new_row_df], ignore_index = True)

        print("******************DONE**********************************************")

    print(df)
    return df
    
