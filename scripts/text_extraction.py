import pymupdf4llm

md = pymupdf4llm.to_markdown('data/raw/Language_model_for_few_shot_learners.pdf')

print(md[:1000])