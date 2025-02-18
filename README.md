# Customer-Support-Bot

## Why choose this project?

You are building an e-commerce site and you want your application to use AI to help improve user experience by using LLM to support users: Respond to users, Look up product information, Suggest products, Request to add products to your cart. This repository will provide a solution microservice for that by providing API using LLM to extract user requests and then divide them into agents according to different tasks to perform.

## Installation

### 1. Ollama

Ollama is a powerful framework that allows you to run and manage large language models (LLMs) locally on your machine, making it easy to integrate various open-source models like Deepseek-R1, Llama3, Mistral, and others into your applications.

In this project, I used Llama3.1:8B to extract user's specific requirements from their messages and Deepseek-R1 for reasoning and answering based on context.
```bash
$ curl -fsSL https://ollama.com/install.sh | sh
$ ollama pull llama3.1:8b
$ ollama pull deepseek-r1
```

### E-commerce Database

Then you need to prepare yourself a simple ecommerce database with the structure below:

![database_architecture](https://github.com/NGrey9/Customer-Support-Bot/blob/main/assets/erd.png)

and use `mongodb` to store chat history information between user and agent


### Dependencies

First, clone the repository:

```bash
$ git clone https://github.com/NGrey9/Customer-Support-Bot.git
$ cd Customer-Support-Bot
```

Install required packages:

```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

You need to create a `.env` file similar to `env_template.txt`

## Implement

Run below command to deploy the service:

```bash
$ python main.py
```

The service will run on `http://localhost:8088/chat`

## Required Input

The input of this endpoint is:

```json
{
  "session_id": "ai12jZ09UUp34",
  "user_id": "usr_00001",
  "message": "Can you tell me some informations about the product named 'Apple iPhone 16 Pro Max'"
}
```
