
# Chemical Reaction Predictor

## Description
The **Chemical Reaction Predictor** is a machine learning-based web application built with Streamlit that predicts the type of chemical reaction, the main product, and by-products based on the input reactants and temperature. The application uses several machine learning models such as K-Nearest Neighbors (KNN), Logistic Regression, and Decision Trees to make predictions. It also provides a visualization of the reaction temperature input and an easy-to-use interface for users to interact with.

## Features
- Predicts the **reaction type** based on two reactants.
- Predicts the **main product** formed from the reaction.
- Predicts the **primary by-product** of the reaction.
- Allows users to input the reactants and temperature for prediction.
- Visualizes the input temperature for better insight.
- Easy-to-use Streamlit interface with a sidebar for user input and information display.

## Requirements
To run this project, you need to install the following Python packages:

- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install these dependencies using `pip`:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

## Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/MohnishE/chemical-reaction-predictor.git
   ```

2. Navigate to the project directory:
   ```bash
   cd chemical-reaction-predictor
   ```

3. Install the required dependencies:
   ```bash
   python3 -m venv RSP
   source RSP/bin/activate
   pip install -r requirements.txt
   ```

4. Ensure that the dataset `data.csv` is present in the project directory. This dataset contains information about reactants, reaction types, products, and by-products, which are essential for training the models.

## Usage

To run the Streamlit application locally, use the following command:

```bash
streamlit run app.py
```

This will open the application in your default web browser, where you can input reactants and temperature values to get predictions.

### Input Details
- **Reactant 1:** Select the first reactant from the dropdown list.
- **Reactant 2:** Select the second reactant from the dropdown list.
- **Temperature:** Use the slider to input the temperature in Celsius (range from 0°C to 600°C).

### Output
- **Predicted Reaction:** The predicted reaction type (e.g., Combustion, Synthesis, etc.).
- **Main Product Formed:** The main product expected to form from the reaction.
- **Primary By-Product:** The primary by-product of the reaction.
- **Suggested Temperature Range:** A recommended temperature range for the reaction (this can be adjusted based on the dataset and chemical reaction properties).

## Project Structure

```
chemical-reaction-predictor/
│
├── app.py                  # Main Streamlit application file
├── data.csv                # Dataset with reaction data (Reactants, Reaction Types, Products, By-Products)
├── requirements.txt        # List of required Python packages
├── README.md               # Project documentation
└── .gitignore              # Git ignore file
```

## Model Details
- **K-Nearest Neighbors (KNN):** Used for predicting the type of chemical reaction based on reactants.
- **Logistic Regression:** Used for predicting the main product formed from the reaction.
- **Decision Tree:** Used for predicting the primary by-product formed during the reaction.

These models are trained using a dataset (`data.csv`), which contains labeled data for chemical reactions.

## Contributing

Contributions are welcome! If you would like to improve this project, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The machine learning models used in this project were trained using `scikit-learn`.
- The web application is built using `Streamlit`.

---

If you have any questions or suggestions, feel free to open an issue or contact the repository owner.
