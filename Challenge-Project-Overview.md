---

> ## Challenge Advisor: Update & Finalize Your Project Overview
>
> > 💡 **These grey text instructions are just for you, the team's Challenge Advisor; please delete them once you have completed the steps below.**
>
> We've pre-populated this Challenge Project Overview page — which is what will be shared with your Break Through Tech student team in August — using the details from your submission form. You should have received an email inviting you to join this repo as a Collaborator, enabling you to add files and make edits.
> 
> In order for your project to be finalized and assigned to a team, please:
> 1. **Review all sections below** and update or expand any content as needed, making sure to address the SME Feedback in the section immediately below. Look for square brackets to find the places below that require additional inputs from you (e.g., "About [Company / Org Name]").
> 2. **Add your dataset** to the [data folder](data) in this repo.
> 3. **Close the Issue assigned to you in this repo** to let us know that you have made your edits and the overview page is ready for final review. You can do this by going to the _Issues_ tab in the top left section of the menu above, add a comment that says "CA review complete", and click the button to Close the Issue. 
>
> If you're unfamiliar with how to edit a page like this in GitHub, check out [this tutorial](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/handson/edit-readme.html) for a quick overview (start with step 2 and only edit this page), and [this guide](https://ubc-lib-geo.github.io/gis-workshop-waml-template/content/markdown.html) on how to use Markdown to compose text.
>
>
> ❌ Remember that this is a public repo. Do NOT include: Proprietary data, PII, API keys, credentials, or anything confidential.

---

## 📋 BTT Internal Evaluation Notes
*(This section is for BTT staff and CAs only — remove before sharing with students)*

### Technical Vetting
| Check | Status | Notes |
| :--- | :--- | :--- |
| Python Compatibility | 🟢 | Project utilizes standard NLP ecosystem (HuggingFace Transformers, PyTorch, Scikit-learn), which is fully compatible with Google Colab. |
| Data Readiness | 🟢 | Bitext and CLINC150 are well-documented, standard benchmark datasets for intent classification. |
| Resource Check | 🟡 | While Free-tier Colab is requested, fine-tuning Phi-2 or DistilBERT can hit VRAM limits. Reliance on external LLM APIs for teacher labeling incurs costs and potential quota issues. |

### Internal Scores
- **Student Fit Score:** 7/10
- **Technical Depth Score:** 8/10
- **Overall Recommendation:** REVISE

### Advisor Feedback Draft
This project offers a high-impact exploration of model distillation, which is a vital skill in modern Enterprise AI. To succeed within the 12-week BTT timeline, I recommend two adjustments: 1) Reduce the synthetic dataset size to 5k-10k high-quality samples to stay within budget/time, and 2) Use a local open-source teacher model (e.g., Llama-3-8B via Ollama or HuggingFace) for label generation to remove dependency on proprietary API keys. 

---

# Domain-Specific Theme Labeling via Parameter-Efficient Fine-Tuning (PEFT) on SLM

**Company / Org:** Automation Anywhere  
**Challenge Advisor:** Archan Dutta, archanduttads@gmail.com  
**AI Studio Coach:** Shaun Figueiro, shaun.figueiro@breakthroughtech.org          
**Program:** Break Through Tech AI Studio - Fall 2026  

---

## 🏢 About Automation Anywhere
Automation Anywhere is a global leader in Intelligent Automation, specializing in Robotic Process Automation (RPA) and AI-driven business solutions. The company empowers organizations to scale their operations through digital transformation, focusing on deploying intelligent software bots to handle complex business processes efficiently.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use datasets (preferably in IT/HR/CX) and ML technique named distillation (using LLMs to teach SLMs) to label topics and then compare the performance of the LLM and SLM. In Enterprise AI, the Distilled SLM may result in a model that matches LLM-level accuracy at lower cost and lower latency. One way to do so is by using Parameter-Efficient Fine-Tuning (PEFT) - LoRA or QLoRA.

### Success Criteria
SLM achieves reasonable accuracy on theme labeling when compared to Frontier LLM.
SLM reduces cost-per-inference when compared to Frontier LLM.
SLM reduces in inference latency when compared to Frontier LLM.
Demo: real-time side-by-side output comparing LLM vs. SLM on live ticket input.

NOTE: The Success metrics could change as the team converges on the project scope

### Stretch Goals
Based on the results of the experiments, write a Research Paper.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Duration | Milestone | Key Activities |
|-------|-----------|----------------|
| **2 weeks - Ends on Sep 15th** | Business Problem and Code Setup | Understand the Business Problem. <br> High-level Project Plan. <br> Team Task Distribution. <br> Environment and Code Setup - Run Smoke Test. |
| **6 weeks - End on Oct 31st** | Planning, Fine-Tuning SLMs and Experimentation] | Detailed Project Planning. <br> Understanding the Pipeline.  <br> Run Fine-Tuning Experiments on Smaller Dataset. <br> Run Fine-Tuning Experiments on Larger Dataset.  <br> Change Implementation as needed - LoRA, QLoRA, Prompt Engineering, Metrics etc.  <br> Evaluation of experiments. |
| **2 weeks - Ends on Nov 15th** | Analysis, Findings and Discussion | Analyze the results of experiments and highlight observations and findings. <br> Discussion on Performance Dimension - Latency, Cost, Quality, Generalizability.|
| **2 weeks - End of Nov 31st** | Storytelling and Presentation | Consolidate Results. <br> Internal Demo. <br> Prepare Presentation for BTT Demo |

NOTE: The Milestones could change as the team converges on the project scope

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** bitext/Bitext-customer-support-llm-chatbot-training-dataset    
**Format:** CSV/ TSV,JSON,Parquet,Excel (.xlsx)     
**Size:** 1gb to 5gb  
**Location:** https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

### Key Details
-[TBD]  

---

## 🛠️ Suggested Approach

**ML Problem Type:** Natural Language Processing (NLP), Large Language Models (LLMs)/ Generative AI / Parameter-Efficient Fine-Tuning / Distillation / 

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]

**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]
  
---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
