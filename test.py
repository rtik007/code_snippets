import asyncio
import pandas as pd
from langchain_google_vertexai import ChatVertexAI

# Your existing synchronous function
def generate_synthetic_data(df, llm, debug=False):
    # Implementation of your function (synchronous)
    pass

async def generate_synthetic_data_async(df, llm, debug=False):
    # Run the synchronous function in a separate thread
    return await asyncio.to_thread(generate_synthetic_data, df, llm, debug)

async def main():
    input_df = pd.DataFrame(data)
    llm = ChatVertexAI(
        model_name=MODEL,
        project=PROJECT_ID,
        temperature=1,
        max_tokens=None,
        max_retries=2,
        stop=None,
        credentials=creds,
        location=LOCATION,
    )

    batch_size = 5
    tasks = []
    for start_idx in range(0, len(input_df), batch_size):
        batch_df = input_df.iloc[start_idx : start_idx + batch_size]
        tasks.append(generate_synthetic_data_async(batch_df, llm, debug=False))

    # Run all batches concurrently
    results = await asyncio.gather(*tasks)

    # Combine all batch outputs
    output_dfs = [result[0] for result in results]  # assuming first return is output_df
    llm_calls_output_dfs = [result[1] for result in results]  # assuming second return

    output_df = pd.concat(output_dfs)
    llm_calls_output_df = pd.concat(llm_calls_output_dfs)

    print(output_df)
    print(llm_calls_output_df)

# Run the async main function
asyncio.run(main())

