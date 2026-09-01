"""
Golden evaluation dataset.

Honesty note on scope: the original project spec called for 100-300 test
questions. This is 11 — a deliberate, documented scope reduction for a
solo one-week build, not a hidden gap. Every question here is traced to
content this system has already returned correctly during manual testing
this week (Days 3-5), so the ground truth is verified, not guessed.
Expanding this set is natural v2 follow-up work — the evaluation harness
itself (run_eval.py, metrics.py) already scales to any dataset size
without changes.

Each case:
- question: what gets sent to /query
- expected_source_document: filename that SHOULD appear in the retrieved
  sources for this to count as a retrieval success. Document-level, not
  chunk-level — chunk IDs aren't stable across re-ingestion, so this is a
  deliberately more robust (if coarser) ground truth signal.
- expected_answer_keywords: substrings that should appear in a correct
  generated answer (case-insensitive). Fast, free faithfulness check —
  the honest limitation is that it's just presence-of-substring, not
  actual semantic correctness.
- expected_answer_summary: a plain-language description of what a
  correct answer should say, used by the LLM-as-judge evaluation
  (llm_judge.py) — a stronger, complementary faithfulness signal added
  as a v2 follow-up. Both checks run and are reported side by side, not
  one replacing the other — a case where they disagree is itself a
  useful finding.
- unanswerable: True for the one deliberate negative test case — this
  question has NO answer anywhere in the corpus, and the correct system
  behavior is to say so, not hallucinate. This is the same test we ran
  live on Day 5 ("What is the capital of France?") formalized into the
  suite so it's checked on every run, not just once by hand.
"""

GOLDEN_DATASET = [
    {
        "id": "resume_dynamics365",
        "question": "What experience does this person have with Dynamics 365?",
        "expected_source_document": "Shagun_Yadav_ATS_Resume_MSD.pdf",
        "expected_answer_keywords": ["Dynamics 365"],
        "expected_answer_summary": "The answer should mention hands-on experience with Microsoft Dynamics 365 CE, including using it alongside Dataverse, Power Pages, and Power Automate in enterprise business applications.",
    },
    {
        "id": "resume_sonarqube",
        "question": "What security tools were used for remediation?",
        "expected_source_document": "Shagun_Yadav_ATS_Resume_MSD.pdf",
        "expected_answer_keywords": ["SonarQube"],
        "expected_answer_summary": "The answer should name Checkmarx and SonarQube as the security tools used, in the context of remediating approximately 80% of application security findings.",
    },
    {
        "id": "resume_verc_revenue",
        "question": "How much annual revenue does the VERC platform enable?",
        "expected_source_document": "Shagun_Yadav_ATS_Resume_MSD.pdf",
        "expected_answer_keywords": ["3.5"],
        "expected_answer_summary": "The answer should state that the VERC platform enables approximately $3.5 billion in annual revenue.",
    },
    {
        "id": "resume_career_progression",
        "question": "What was the career progression at Visa, from first role to most recent?",
        "expected_source_document": "Shagun_Yadav_ATS_Resume_MSD.pdf",
        "expected_answer_keywords": ["Software Development Intern", "Senior Software Engineer"],
        "expected_answer_summary": "The answer should describe a progression starting as Software Development Intern, moving to Software Engineer, and most recently Senior Software Engineer.",
    },
    {
        "id": "resume_raahee",
        "question": "What technology was used at the Raahee internship?",
        "expected_source_document": "Shagun_Yadav_ATS_Resume_MSD.pdf",
        "expected_answer_keywords": ["React"],
        "expected_answer_summary": "The answer should mention React.js was used to build the startup's web application during the Raahee internship.",
    },
    {
        "id": "research_rank1_accuracy",
        "question": "What was the Rank-1 accuracy achieved by the fine-tuned Vision Transformer?",
        "expected_source_document": "Group_7_Major_Project_Report_C2CL.pdf",
        "expected_answer_keywords": ["98.49"],
        "expected_answer_summary": "The answer should state the fine-tuned Vision Transformer achieved 98.49% Rank-1 Accuracy, and ideally note it outperformed the Sequential CNN model.",
    },
    {
        "id": "research_auc_roc",
        "question": "What AUC-ROC score did the fingerprint recognition model achieve?",
        "expected_source_document": "research paper ieee.pdf",
        "expected_answer_keywords": ["99.99"],
        "expected_answer_summary": "The answer should state the model achieved a 99.99% AUC-ROC score.",
    },
    {
        "id": "research_equal_error_rate",
        "question": "What was the Equal Error Rate of the fine-tuned model?",
        "expected_source_document": "research paper ieee.pdf",
        # Accepts either phrasing the model might use for the same value
        # (0.40% == 0.0040 as a decimal fraction) — a real answer scored
        # a false failure here on the first eval run because the model
        # said "0.0040" and the keyword check only looked for "0.40".
        # Documented rather than silently patched: general numeric-
        # equivalence checking (parsing both sides as numbers and
        # comparing) would be a more robust fix, left as a known
        # follow-up rather than over-built for one case. The LLM-judge
        # check added alongside this doesn't have this brittleness — a
        # judge correctly treats "0.40%" and "0.0040" as equivalent,
        # which is itself a useful side-by-side comparison to make.
        "expected_answer_keywords": ["0.40", "0.004"],
        "expected_answer_summary": "The answer should state the Equal Error Rate was 0.40% (equivalently expressed as 0.0040 as a decimal).",
    },
    {
        "id": "research_dataset_used",
        "question": "What fingerprint database was used to evaluate the models?",
        "expected_source_document": "Group_7_Major_Project_Report_C2CL.pdf",
        "expected_answer_keywords": ["PolyU"],
        "expected_answer_summary": "The answer should name the PolyU fingerprint database as what was used for evaluation.",
    },
    {
        "id": "visa_visanet",
        "question": "What network does Visa operate?",
        "expected_source_document": "test.txt",
        "expected_answer_keywords": ["VisaNet"],
        "expected_answer_summary": "The answer should name VisaNet as the network Visa operates.",
    },
    {
        "id": "negative_unanswerable",
        "question": "What is the capital of France?",
        "unanswerable": True,
    },
]
