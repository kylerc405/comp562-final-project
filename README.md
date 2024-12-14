# CS Role Career Predictor

## [Live Demo](https://comp562-final.streamlit.app/)

This project uses machine learning to predict whether someone is likely to be in a Computer Science (CS) role based on their skills and education level. The model analyzes patterns in skill sets and educational background to determine the likelihood of a person working in a CS-related position.

## Features

- **Skill Selection**: Choose from a curated list of relevant professional skills
- **Education Level Input**: Specify your highest CS education level
- **Real-time Prediction**: Get instant feedback on your career trajectory
- **Confidence Scores**: View the model's confidence in its prediction
- **Visual Results**: See probability distributions through interactive charts

## How It Works

1. **Data Collection**: The model was trained on a dataset containing:
   - Individual skill profiles
   - Educational backgrounds
   - Current job roles (CS vs non-CS)

2. **Feature Processing**:
   - Skills are processed using MultiLabelBinarizer
   - Dimensionality reduction is applied using TruncatedSVD
   - Education levels are encoded numerically

3. **Model**:
   - Uses Random Forest Classifier
   - Combines skill features with education level
   - Provides probability scores for predictions

## Usage

1. Visit [https://comp562-final.streamlit.app/](https://comp562-final.streamlit.app/)
2. Select your skills from the dropdown menu
3. Choose your highest CS education level
4. Click "Predict" to see your results

## Education Levels

- **None** (0): No formal CS education
- **Associate's Degree** (1): 2-year degree in CS
- **Bachelor's Degree** (2): 4-year degree in CS
- **Master's Degree** (3): Graduate degree in CS
- **PhD** (4): Doctorate in CS

## Interpretation of Results

- **Prediction**: Whether you're likely to be in a CS role
- **Confidence**: The model's certainty in its prediction
- **Probability Distribution**: Relative likelihood of CS vs non-CS roles

## Technical Details

- **Framework**: Streamlit
- **ML Libraries**: scikit-learn
- **Key Components**:
  - Random Forest Classifier
  - MultiLabelBinarizer for skill encoding
  - TruncatedSVD for dimensionality reduction

## Limitations

- Predictions are based on historical data patterns
- The model considers only listed skills and education level
- Other factors like experience duration and specific roles aren't considered
- Results should be interpreted as guidance rather than definitive career advice

## Future Improvements

- Addition of experience duration as a feature
- More granular role categorization
- Industry-specific skill weightings
- Incorporation of geographical factors
- More detailed educational background analysis
