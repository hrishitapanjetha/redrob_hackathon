# redrob_hackathon
# Resume Matching Engine

Built this for the **Redrob AI Campus Hackathon**. The task was to match 10 student resumes against 3 job descriptions (from Korean tech companies — Kakao, Naver, Line) and return the top 3 candidates per JD.

## What it does

Takes noisy resume skills (things like "Pyhton", "MachineLearning", "kubernates"), cleans them up using an alias map, then does TF-IDF + cosine similarity to find the best resume matches for each JD.

Standard library only — no numpy, no pandas, no sklearn. Just `math` and basic Python.

## Results

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```


## Approach

- **Normalize**: lowercase, split on commas, look up each token in `SKILL_ALIASES`, drop anything unknown.
- **Deduplicate**: each canonical skill appears at most once per resume.
- **Vocab**: union of all canonical skills across the 10 resumes, sorted alphabetically. Built from resumes only.
- **TF-IDF**: TF = 1/N (since each skill appears once after dedup), IDF = ln(10/df), no smoothing.
- **JD vectors**: binary, 1 if the JD has the skill (after the same normalization) AND it's in vocab, else 0.
- **Similarity**: cosine, with alphabetical tiebreak on candidate name.



## Built with

Python 3, Google Colab, and Redrob AI as the coding assistant.
