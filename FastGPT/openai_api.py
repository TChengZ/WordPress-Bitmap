import asyncio
import logging
import time
from typing import List, Literal, Optional, Union

import chatglm_cpp
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field#, computed_field
#from pydantic_settings import BaseSettings
from sse_starlette.sse import EventSourceResponse

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import tiktoken

logging.basicConfig(level=logging.INFO, format=r"%(asctime)s - %(module)s - %(levelname)s - %(message)s")


class Settings(object):
    model: str = "/Users/chenzujie/work/Ai/chatglm.cpp/chatglm3-ggml-q8.bin";
    num_threads: int = 0


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class DeltaMessage(BaseModel):
    role: Optional[Literal["system", "user", "assistant"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "default-model"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    stream: bool = False
    max_tokens: int = Field(default=2048, ge=0)

    model_config = {
        "json_schema_extra": {"examples": [{"model": "default-model", "messages": [{"role": "user", "content": "你好"}]}]}
    }


class ChatCompletionResponseChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    #@computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl"
    model: str = "default-model"
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: Union[List[ChatCompletionResponseChoice], List[ChatCompletionResponseStreamChoice]]
    usage: Optional[ChatCompletionUsage] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "chatcmpl",
                    "model": "default-model",
                    "object": "chat.completion",
                    "created": 1691166146,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "你好👋！我是人工智能助手 ChatGLM2-6B，很高兴见到你，欢迎问我任何问题。"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 17, "completion_tokens": 29, "total_tokens": 46},
                }
            ]
        }
    }


settings = Settings()
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
pipeline = chatglm_cpp.Pipeline(settings.model)
lock = asyncio.Lock()


def stream_chat(messages, body):
    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(role="assistant"))],
    )

    for chunk in pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
        num_threads=settings.num_threads,
        stream=True,
    ):
        yield ChatCompletionResponse(
            object="chat.completion.chunk",
            choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(content=chunk.content))],
        )

    yield ChatCompletionResponse(
        object="chat.completion.chunk",
        choices=[ChatCompletionResponseStreamChoice(delta=DeltaMessage(), finish_reason="stop")],
    )


async def stream_chat_event_publisher(history, body):
    output = ""
    try:
        async with lock:
            for chunk in stream_chat(history, body):
                await asyncio.sleep(0)  # yield control back to event loop for cancellation check
                output += chunk.choices[0].delta.content or ""
                yield chunk.model_dump_json(exclude_unset=True)
        logging.info(f'prompt: "{history[-1]}", stream response: "{output}"')
    except asyncio.CancelledError as e:
        logging.info(f'prompt: "{history[-1]}", stream response (partial): "{output}"')
        raise e


@app.post("/v1/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest) -> ChatCompletionResponse:
    if not body.messages:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty messages")

    messages = [chatglm_cpp.ChatMessage(role=msg.role, content=msg.content) for msg in body.messages]

    if body.stream:
        generator = stream_chat_event_publisher(messages, body)
        return EventSourceResponse(generator)

    max_context_length = 512
    output = pipeline.chat(
        messages=messages,
        max_length=body.max_tokens,
        max_context_length=max_context_length,
        do_sample=body.temperature > 0,
        top_p=body.top_p,
        temperature=body.temperature,
    )
    logging.info(f'prompt: "{messages[-1].content}", sync response: "{output.content}"')
    prompt_tokens = len(pipeline.tokenizer.encode_messages(messages, max_context_length))
    completion_tokens = len(pipeline.tokenizer.encode(output.content, body.max_tokens))

    return ChatCompletionResponse(
        object="chat.completion",
        choices=[ChatCompletionResponseChoice(message=ChatMessage(role="assistant", content=output.content))],
        usage=ChatCompletionUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "owner"
    permission: List = []


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "object": "list",
                    "data": [{"id": "gpt-3.5-turbo", "object": "model", "owned_by": "owner", "permission": []}],
                }
            ]
        }
    }


@app.get("/v1/models")
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id="gpt-3.5-turbo")])

embeddings_model = SentenceTransformer('/Users/chenzujie/work/Ai/m3e-base', device='cpu')

class EmbeddingRequest(BaseModel):
    input: List[str]
    model: str

class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens

def expand_features(embedding, target_length):
    poly = PolynomialFeatures(degree=2)
    expanded_embedding = poly.fit_transform(embedding.reshape(1, -1))
    expanded_embedding = expanded_embedding.flatten()
    if len(expanded_embedding) > target_length:
        # 如果扩展后的特征超过目标长度，可以通过截断或其他方法来减少维度
        expanded_embedding = expanded_embedding[:target_length]
    elif len(expanded_embedding) < target_length:
        # 如果扩展后的特征少于目标长度，可以通过填充或其他方法来增加维度
        expanded_embedding = np.pad(expanded_embedding, (0, target_length - len(expanded_embedding)))
    return expanded_embedding


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    
    
    # 计算嵌入向量和tokens数量 
    embeddings = [embeddings_model.encode(text) for text in request.input]

    # 如果嵌入向量的维度不为1536，则使用插值法扩展至1536维度 
    embeddings = [expand_features(embedding, 1536) if len(embedding) < 1536 else embedding for embedding in embeddings]
	
    # Min-Max normalization
    embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]

    # 将numpy数组转换为列表
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in request.input)
    total_tokens = sum(num_tokens_from_string(text) for text in request.input)

    
    response = {
        "data": [
            {
                "embedding": embedding,
                "index": index,
                "object": "embedding"
            } for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
    }
    
    return response



