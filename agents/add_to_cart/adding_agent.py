import os
from dotenv import load_dotenv

import requests
from sqlalchemy import text
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


from agent import Agent
from .adding_request import AddingRequest
load_dotenv()

AGENT_ADDING = os.environ["AGENT_ADDING"]
ADDING_ENDPOINT = os.environ["ADDING_ENDPOINT"]


class AddingAgent(Agent):

    prompt = """
    You are an agent helping customers add products to their shopping cart. 
    Based on the information below, respond to customers:

    Context: {context}
    User query: {query}

    Notify the customer that the product by the name provided by the customer has been added to the customer's shopping cart.
    If the status is not added to cart, you need to notity to the customer that the product not added into their cart and require them to try again.
    If the product is not found, notify the customer that the product they want to add to the cart cannot be found and ask the customer to provide the name of the product they want to buy again.
    """

    model_name = AGENT_ADDING
    adding_endpoint = ADDING_ENDPOINT

    def __init__(self, sql_database_api):
        super().__init__(agent_name="AddingAgent")
        self.database_api = sql_database_api
        self.create_chain()

    def get_context(self, context_input: dict):
        user_id = context_input.get("user_id")
        product_name = context_input.get("product_name")
        product_id = self.get_product_id(
            product_name=product_name)
        if product_id:
            print("product_id: ", product_id)
            status = self.send_request(user_id=user_id, product_id=product_id)
            if status:
                print("aaaa")
                return f"Product: {product_name}\nStatus: be added to customer cart successfully."
            else:
                return f"Product: {product_name}\nStatus: be not added to customer cart. Please try again."
        else:
            return f"Product: {product_name}\nStatus: not found. "

    def create_chain(self):
        try:
            self.chain = RunnableParallel(
                query=RunnablePassthrough(),
                context=lambda x: self.get_context(x)
            ) | self.prompt_template | self.llm | StrOutputParser()
        except Exception as e:
            raise e

    def get_product_id(self, product_name):
        with self.database_api.engine.connect() as conn:
            try:
                results = conn.execute(
                    text(
                        f"SELECT id FROM products WHERE name = \"{product_name}\"".strip(
                            "'")
                    )
                )
                data = [{"id": row[0]} for row in results]
                return data[0]["id"]
            except Exception as e:
                return None

    def send_request(self, user_id, product_id) -> bool:
        payload = AddingRequest(user_id=user_id,
                                product_id=product_id)
        try:
            response = requests.post(self.adding_endpoint, payload)
            return True
        except Exception as e:
            return False

    def answer(self, user_id, product_name):
        try:
            response = self.chain.invoke(
                {"user_id": user_id, "product_name": product_name})
            message = self.postprocess_message(response)
            return message
        except Exception as e:
            raise e
