# Budget Recommendation with Graph Features

an intuitive and powerful tool designed to assist in optimizing and recommending budget allocations based on key metrics such as efficiency, influence, and network centrality. Developed with **Streamlit**, this app provides interactive analysis, detailed visualizations, and actionable insights to support data-driven decision-making.

## Features

- **Dynamic Budget Recommendations**: Leverages advanced optimization techniques to recommend budget adjustments while maintaining total budget consistency.
- **Interactive Adjustments**: Users can fine-tune recommendations directly in the app and see updated suggestions in real-time.
- **Data Cleaning & Validation**: Ensures accurate and reliable processing of input data with robust error handling.
- **Visualization Tools**: Provides comprehensive visual insights into budget changes, including:
  - Bar charts for budget comparisons.
  - Pie charts for budget distribution.
  - Percentage change visualizations.
- **Detailed Explanations**: Generates insights into why adjustments are recommended, linking them to key metrics such as PageRank, efficiency ratio, and degree centrality.

## Tech Stack

- **Python**: Core logic and data processing.
- **Streamlit**: Interactive web app framework.
- **Pandas & NumPy**: Data manipulation and numerical computations.
- **SciPy**: Optimization routines for budget allocation.
- **Plotly**: Advanced visualization for interactive and high-quality charts.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/budget-recommendation-app.git
   ```
2. Navigate to the project directory:
   ```bash
   cd budget-recommendation-app
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload the **Analysis CSV** containing key metrics for each budgetary unit (e.g., `pagerank`, `efficiency_ratio`, etc.).
2. Upload the **Original Budget CSV** detailing current budget allocations.
3. View initial recommendations, fine-tune budgets interactively, and analyze updated recommendations with comprehensive visualizations.
4. Export the results for further analysis or reporting.

## Input Data Format

### Analysis CSV
- Must include the following columns:
  - `name`: Identifier for budgetary units.
  - `pagerank`, `efficiency_ratio`, `degree_centrality`, `influence_efficiency`: Metrics for analysis.

### Original Budget CSV
- Must include the following columns:
  - `BUDGETARY_UNIT`: Matching identifier for budgetary units.
  - `AMOUNT`: Original budget allocation.

## Example Workflow

1. **Upload Files**: Provide the analysis and budget data files in `.csv` format.
2. **View Recommendations**: Explore suggested adjustments and their justifications.
3. **Adjust & Update**: Make manual adjustments to the budgets and visualize the impact.
4. **Download Results**: Export the recommendations for presentation or implementation.

## Contribution

Contributions are welcome! If you find issues or have feature requests, feel free to open an issue or submit a pull request.
