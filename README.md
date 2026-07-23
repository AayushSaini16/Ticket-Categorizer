# Auto Email / Ticket Categorizer

AI/ML Intern Assessment submission - Fobes Skill Itech

## What this project does

Reads a support ticket (subject + body text) and predicts which department it
should go to: **Billing, Technical, HR or General**. Along with the category it
also gives a confidence score, flags low-confidence tickets for human review,
and adds a simple urgent/normal priority tag.

## Dataset

The dataset used is a small **dummy dataset that I created myself** for this
assessment (`data/build_dataset.py`), 80 tickets in total, 20 per category. No
real or public dataset has been used, as per the assessment instructions.

## Files in this folder

| File | Description |
|---|---|
| `data/build_dataset.py` | Script that creates the dummy dataset and saves it as `data/tickets.csv` |
| `data/tickets.csv` | The dummy dataset (subject, body, category) |
| `ticket_categorizer.py` | Main script - cleaning, TF-IDF, model training, evaluation, predictions |
| `ticket_categorizer.ipynb` | Same work as a Jupyter notebook, with explanations |
| `confusion_matrix.png` | Confusion matrix image generated after running the script |
| `sample_predictions.csv` | Predictions on the 5 new sample tickets |
| `requirements.txt` | Python packages needed to run this |

## How to run

```bash
pip install -r requirements.txt
python ticket_categorizer.py
```

This will print the accuracy, classification report and confusion matrix in
the terminal, save `confusion_matrix.png` and `sample_predictions.csv`, and
print predictions for 5 new sample tickets.

To try the live CLI demo, open `ticket_categorizer.py`, uncomment the last
line (`cli_demo(...)`) and run the script again. It will let you type a ticket
and see the predicted category instantly.

## Approach (short version)

1. Combined ticket subject + body into a single text field
2. Cleaned text - lowercase, removed punctuation/numbers, removed stopwords
3. Converted text to TF-IDF vectors
4. Trained a Multinomial Naive Bayes classifier
5. Evaluated using accuracy, precision/recall and a confusion matrix
6. Predicted category on 5 new tickets, along with a confidence score
7. If confidence is below 60%, the ticket is marked as "needs human review"
   instead of being auto-assigned
8. Added a simple keyword-based priority tag (Urgent/Normal)

## Known limitations

- Dataset is small (80 tickets), so accuracy numbers are only indicative, not
  something to rely on for a real system
- Confidence scores are on the lower side for Naive Bayes on a small dataset,
  even when the predicted category is correct
- The 60% review threshold is a fixed number I picked, not something tuned
  using a validation set

## Reflection

If I had more time or a bigger dataset, I would want to try Logistic
Regression alongside Naive Bayes and compare which one gives better and more
confident predictions. I would also collect a bigger and more varied set of
real ticket phrasing so the model has more to learn from, and tune the
review-threshold value properly using a validation set instead of picking it
manually.

