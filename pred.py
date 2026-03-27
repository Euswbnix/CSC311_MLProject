import sys
import csv
import re
import numpy as np
import pandas as pd


import os
_DIR = os.path.dirname(os.path.abspath(__file__))
_params = np.load(os.path.join(_DIR, 'model_params.npz'), allow_pickle=True)

COEF = _params['coef']                 # (3, 302)
INTERCEPT = _params['intercept']       # (3,)
CLASSES = _params['classes']           # ['The Persistence of Memory', 'The Starry Night', 'The Water Lily Pond']
SCALER_MIN = _params['scaler_min']     # (22,)
SCALER_RANGE = _params['scaler_range'] # (22,)
FILL_MEDIANS = _params['fill_medians'] # (8,)

# Reconstruct TF-IDF vocabularies {word: index} and idf vectors
TFIDF = {}
for col in ['feeling_desc', 'food', 'soundtrack']:
    keys = _params[f'tfidf_{col}_vocab_keys']
    vals = _params[f'tfidf_{col}_vocab_vals']
    idf = _params[f'tfidf_{col}_idf']
    TFIDF[col] = {
        'vocab': dict(zip(keys, vals.astype(int))),
        'idf': idf,
        'n_features': len(idf),
    }



# Preprocessing helper functions (pure python + numpy)
# Column name mapping: original CSV header → short name
COLUMN_MAP = {
    'On a scale of 1–10, how intense is the emotion conveyed by the artwork?': 'emotion_intensity',
    'On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?': 'emotion_intensity',
    'Describe how this painting makes you feel.': 'feeling_desc',
    'This art piece makes me feel sombre.': 'sombre',
    'This art piece makes me feel content.': 'content',
    'This art piece makes me feel calm.': 'calm',
    'This art piece makes me feel uneasy.': 'uneasy',
    'How many prominent colours do you notice in this painting?': 'num_colours',
    'How many objects caught your eye in the painting?': 'num_objects',
    'How much (in Canadian dollars) would you be willing to pay for this painting?': 'willing_to_pay',
    'If you could purchase this painting, which room would you put that painting in?': 'room',
    'If you could view this art in person, who would you want to view it with?': 'view_with',
    'What season does this art piece remind you of?': 'season',
    'If this painting was a food, what would be?': 'food',
    'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.': 'soundtrack',
}

LIKERT_MAP = {
    '1 - Strongly disagree': 1.0,
    '2 - Disagree': 2.0,
    '3 - Neutral/Unsure': 3.0,
    '4 - Agree': 4.0,
    '5 - Strongly agree': 5.0,
}


def clean_price(val):
    """Extract numeric dollar amount from messy text."""
    if pd.isna(val) or str(val).strip() == '':
        return np.nan
    text = str(val).lower().strip()
    multiplier = 1
    if 'billion' in text:
        multiplier = 1_000_000_000
    elif 'million' in text:
        multiplier = 1_000_000
    text = text.replace('$', '').replace(',', '').replace('cad', '')
    text = text.replace('dollars', '').replace('dollar', '')
    text = text.replace('bucks', '').replace('buck', '')
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    match = re.search(r'(\d+\.?\d*)', text)
    if not match:
        return np.nan
    value = float(match.group(1))
    if multiplier > 1 and value < 1000:
        value *= multiplier
    return value if value <= 10_000_000 else np.nan


def tfidf_transform(text, vocab, idf, n_features):
    """
    Compute TF-IDF vector for a single document.
    Reimplements sklearn's TfidfVectorizer transform (l2-normalized).
    """
    if not isinstance(text, str) or text.strip() == '':
        return np.zeros(n_features)

    # Tokeniz
    tokens = re.findall(r'(?u)\b\w\w+\b', text.lower())
    tf = np.zeros(n_features)
    for token in tokens:
        if token in vocab:
            tf[vocab[token]] += 1
    tfidf_vec = tf * idf
    # L2 normalization
    norm = np.sqrt(np.sum(tfidf_vec ** 2))
    if norm > 0:
        tfidf_vec /= norm
    return tfidf_vec


def preprocess_row(row):
    """
    Convert a single test row (dict or Series) into a 302-dim feature vector.
    """
    ei = row.get('emotion_intensity', np.nan)
    ei = float(ei) if pd.notna(ei) and str(ei).strip() != '' else FILL_MEDIANS[0]
    nc = row.get('num_colours', np.nan)
    nc = float(nc) if pd.notna(nc) and str(nc).strip() != '' else FILL_MEDIANS[1]
    nc = np.clip(nc, 0, 20)
    no = row.get('num_objects', np.nan)
    no = float(no) if pd.notna(no) and str(no).strip() != '' else FILL_MEDIANS[2]
    no = np.clip(no, 0, 20)
    # Willing to pay
    price = clean_price(row.get('willing_to_pay', ''))
    if np.isnan(price):
        price = FILL_MEDIANS[3]
    price = np.clip(price, 0, 1_000_000)
    # Likert scales
    sombre_val = LIKERT_MAP.get(str(row.get('sombre', '')), FILL_MEDIANS[4])
    content_val = LIKERT_MAP.get(str(row.get('content', '')), FILL_MEDIANS[5])
    calm_val = LIKERT_MAP.get(str(row.get('calm', '')), FILL_MEDIANS[6])
    uneasy_val = LIKERT_MAP.get(str(row.get('uneasy', '')), FILL_MEDIANS[7])
    # Season binary
    season_str = str(row.get('season', ''))
    season_spring = 1.0 if 'Spring' in season_str else 0.0
    season_summer = 1.0 if 'Summer' in season_str else 0.0
    season_fall   = 1.0 if 'Fall' in season_str else 0.0
    season_winter = 1.0 if 'Winter' in season_str else 0.0
    # Room binary
    room_str = str(row.get('room', ''))
    room_bedroom    = 1.0 if 'Bedroom' in room_str else 0.0
    room_bathroom   = 1.0 if 'Bathroom' in room_str else 0.0
    room_office     = 1.0 if 'Office' in room_str else 0.0
    room_living     = 1.0 if 'Living room' in room_str else 0.0
    room_dining     = 1.0 if 'Dining room' in room_str else 0.0
    # View with binary
    view_str = str(row.get('view_with', ''))
    view_friends    = 1.0 if 'Friends' in view_str else 0.0
    view_family     = 1.0 if 'Family members' in view_str else 0.0
    view_coworkers  = 1.0 if 'Coworkers/Classmates' in view_str else 0.0
    view_strangers  = 1.0 if 'Strangers' in view_str else 0.0
    view_yourself   = 1.0 if 'By yourself' in view_str else 0.0

    struct_vec = np.array([
        ei, sombre_val, content_val, calm_val, uneasy_val,
        nc, no, price,
        season_spring, season_summer, season_fall, season_winter,
        room_bedroom, room_bathroom, room_office, room_living, room_dining,
        view_friends, view_family, view_coworkers, view_strangers, view_yourself,
    ])

    safe_range = np.where(SCALER_RANGE == 0, 1.0, SCALER_RANGE)
    struct_scaled = (struct_vec - SCALER_MIN) / safe_range
    struct_scaled = np.clip(struct_scaled, 0, 1)

    feeling_vec = tfidf_transform(
        str(row.get('feeling_desc', '')),
        TFIDF['feeling_desc']['vocab'],
        TFIDF['feeling_desc']['idf'],
        TFIDF['feeling_desc']['n_features']
    )
    food_vec = tfidf_transform(
        str(row.get('food', '')),
        TFIDF['food']['vocab'],
        TFIDF['food']['idf'],
        TFIDF['food']['n_features']
    )
    soundtrack_vec = tfidf_transform(
        str(row.get('soundtrack', '')),
        TFIDF['soundtrack']['vocab'],
        TFIDF['soundtrack']['idf'],
        TFIDF['soundtrack']['n_features']
    )

    # Concatenate: struct (22) + feeling (150) + food (50) + soundtrack (80) = 302
    return np.concatenate([struct_scaled, feeling_vec, food_vec, soundtrack_vec])


def predict(x_vec):
    """
    Predict class for a single 302-dim feature vector.
    Logistic regression: scores = x @ W.T + b, then argmax.
    """
    scores = x_vec @ COEF.T + INTERCEPT  # (3,)
    return CLASSES[np.argmax(scores)]


def predict_all(filename):
    """
    Make predictions for the data in filename.

    Parameters
    ----------
    filename : str
        Path to a CSV file with the same columns as the training set.

    Returns
    -------
    list of str
        Predicted painting names for each row.
    """
    df = pd.read_csv(filename)
    df.rename(columns=COLUMN_MAP, inplace=True)

    predictions = []
    for _, row in df.iterrows():
        feature_vec = preprocess_row(row)
        pred = predict(feature_vec)
        predictions.append(pred)

    return predictions


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = 'data/ml_challenge_dataset.csv'

    preds = predict_all(test_file)
    print(f"Total predictions: {len(preds)}")
    print(f"Sample predictions: {preds[:5]}")

    # If running on training data, compute accuracy
    df_check = pd.read_csv(test_file)
    if 'Painting' in df_check.columns:
        true_labels = df_check['Painting'].values
        correct = sum(p == t for p, t in zip(preds, true_labels))
        print(f"Accuracy on this file: {correct}/{len(preds)} = {correct/len(preds):.2%}")