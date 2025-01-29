import streamlit as st
from dotenv import load_dotenv
import Invoiceutil as iu

def main():
    load_dotenv()

    st.set_page_config(page_title="Invoice Extraction Bot")
    st.title("Invoice Extraction Bot ...")
    st.subheader("I can help you extrat invoice data")


    pdf = st.file_uploader("Upload your invoices here, only PDFs allowed", type = ["pdf"], accept_multiple_files=True)

    submit = st.button("Extract Data")

    if submit:
        with st.spinner("Wait while we load your data"):
            df = iu.create_docs(pdf)
            st.write(df.head())

            data_as_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Click here to download data",
                               data_as_csv,
                               "benchmark-tools.csv",
                               "text/csv",
                               key = "download-tools-csv",
                               )
            
            st.success("Hope I was able to help save your time")


if __name__ == "__main__":
    main()