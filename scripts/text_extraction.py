import pymupdf4llm

md = pymupdf4llm.to_markdown('data/raw/attention_is_all_you_need.pdf')

print(md[:1000])