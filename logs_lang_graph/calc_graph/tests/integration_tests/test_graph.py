import pytest

from agent import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    inputs = {
        "messages": [
            {
                "role": "human",
                "content": "Add 3 and 4.",
            }
        ]
    }
    res = await graph.ainvoke(inputs)
    assert res is not None
