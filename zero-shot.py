import dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
###############################################################################
prompt = ChatPromptTemplate.from_template("""
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
""")

class SentimentClassification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text",
                           enum=["positive", "neutral", "negative"])

chain = prompt | llm.with_structured_output(SentimentClassification)
###############################################################################
df = pd.read_csv("data/financial-sentiment-analysis.csv")
df["Classification"] = ""

for i, r in df.iterrows():
    if i % 100 == 0: # pyright: ignore
        print(i)
    inp = r.Sentence
    res = chain.invoke({"input": inp})
    df.loc[i, "Classification"] = res.sentiment # pyright:ignore

df.to_csv("data/zero-shot-results.csv")
