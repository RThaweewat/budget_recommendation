import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go


# Copy the generate_budget_recommendations function here
def generate_budget_recommendations(df, original_budget_df):
	# Merge the analysis dataframe with the original budget dataframe
	merged_df = pd.merge(df, original_budget_df, left_on='name', right_on='BUDGETARY_UNIT', how='left')

	# Normalize the metrics
	scaler = MinMaxScaler()
	metrics = ['pagerank', 'efficiency_ratio', 'degree_centrality', 'influence_efficiency']
	merged_df[metrics] = scaler.fit_transform(merged_df[metrics])

	# Store original metric values for explanation
	original_metrics = merged_df[metrics].copy()

	# Define the objective function to minimize
	def objective(adjustments):
		new_budgets = merged_df['AMOUNT'] * (1 + adjustments)

		# Calculate scores based on new budgets and normalized metrics
		efficiency_score = np.sum((merged_df['efficiency_ratio'] - new_budgets / merged_df['AMOUNT'])**2)
		influence_score = np.sum((merged_df['pagerank'] - new_budgets / merged_df['AMOUNT'])**2)
		centrality_score = np.sum((merged_df['degree_centrality'] - new_budgets / merged_df['AMOUNT'])**2)
		influence_efficiency_score = np.sum((merged_df['influence_efficiency'] - new_budgets / merged_df['AMOUNT'])**2)

		# Combine scores
		total_score = efficiency_score + influence_score + centrality_score + influence_efficiency_score

		# Add a penalty for deviating from the original total budget
		budget_deviation_penalty = (np.sum(new_budgets) - np.sum(merged_df['AMOUNT']))**2 * 1000

		# Add a penalty for not making changes
		no_change_penalty = np.sum(adjustments**2) * 10

		return total_score + budget_deviation_penalty - no_change_penalty

	# Initial guess: small random changes
	initial_adjustments = np.random.uniform(-0.1, 0.1, len(merged_df))

	# Constraints: limit adjustments to ±20%
	bounds = [(-0.2, 0.2) for _ in range(len(merged_df))]

	# Constraint to keep total budget constant
	def budget_constraint(adjustments):
		return np.sum(merged_df['AMOUNT'] * (1 + adjustments)) - np.sum(merged_df['AMOUNT'])

	constraints = ({'type': 'eq', 'fun': budget_constraint})

	# Perform the optimization
	result = minimize(objective, initial_adjustments, method='SLSQP', bounds=bounds, constraints=constraints)

	# Apply the optimized adjustments
	merged_df['recommended_budget'] = merged_df['AMOUNT'] * (1 + result.x)
	merged_df['adjustment'] = result.x

	# Generate detailed explanations based on the adjustments and metrics
	def get_detailed_explanation(row):
		adj = row['adjustment']
		explanation = []

		# Define thresholds for significant differences
		threshold = 0.2  # 20% difference is considered significant

		if abs(adj) > 0.01:  # If there's a noticeable adjustment
			if adj > 0:
				explanation.append(f"Budget increase of {adj*100:.2f}% recommended.")
			else:
				explanation.append(f"Budget decrease of {-adj*100:.2f}% recommended.")

			# Explain based on each metric
			if abs(row['pagerank'] - original_metrics.loc[row.name, 'pagerank']) > threshold:
				if row['pagerank'] > original_metrics.loc[row.name, 'pagerank']:
					explanation.append("The unit's influence in the network has increased.")
				else:
					explanation.append("The unit's influence in the network has decreased.")

			if abs(row['efficiency_ratio'] - original_metrics.loc[row.name, 'efficiency_ratio']) > threshold:
				if row['efficiency_ratio'] > original_metrics.loc[row.name, 'efficiency_ratio']:
					explanation.append("The unit's efficiency has improved.")
				else:
					explanation.append("The unit's efficiency has declined.")

			if abs(row['degree_centrality'] - original_metrics.loc[row.name, 'degree_centrality']) > threshold:
				if row['degree_centrality'] > original_metrics.loc[row.name, 'degree_centrality']:
					explanation.append("The unit's connectivity in the network has increased.")
				else:
					explanation.append("The unit's connectivity in the network has decreased.")

			if abs(row['influence_efficiency'] - original_metrics.loc[row.name, 'influence_efficiency']) > threshold:
				if row['influence_efficiency'] > original_metrics.loc[row.name, 'influence_efficiency']:
					explanation.append("The unit's influence relative to its budget has improved.")
				else:
					explanation.append("The unit's influence relative to its budget has declined.")

		else:
			explanation.append("No significant change recommended based on current metrics.")

		if not explanation[1:]:  # If no specific metric explanations were added
			explanation.append("The adjustment is based on a balanced consideration of all metrics.")

		return " ".join(explanation)

	merged_df['detailed_explanation'] = merged_df.apply(get_detailed_explanation, axis=1)

	return merged_df[['name', 'AMOUNT', 'recommended_budget', 'detailed_explanation', 'adjustment']]


def format_budget(value):
    """Format budget values in billions with 2 decimal places."""
    return f"{value:.2f}B"

def clean_and_convert_amount(df):
    """Clean and convert the AMOUNT column to numeric values."""
    if 'AMOUNT' not in df.columns:
        st.error("Error: 'AMOUNT' column not found in the uploaded file.")
        return None

    # Try to convert directly to numeric
    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')

    # If conversion failed, try cleaning the data
    if df['AMOUNT'].isna().all():
        # Remove any non-numeric characters (except '.' for decimal point)
        df['AMOUNT'] = df['AMOUNT'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')

    # If conversion still failed, raise an error
    if df['AMOUNT'].isna().all():
        st.error("Error: Unable to convert 'AMOUNT' column to numeric values. Please check your data format.")
        return None

    return df

def main():
    st.set_page_config(layout="wide")  # Use wide layout for better visibility
    st.title("Budget Recommendation App")

    # File uploaders
    analysis_file = st.file_uploader("Upload Analysis CSV", type="csv")
    budget_file = st.file_uploader("Upload Original Budget CSV", type="csv")

    if analysis_file and budget_file:
        try:
            analysis_df = pd.read_csv(analysis_file)
            budget_df = pd.read_csv(budget_file)

            # Clean and convert AMOUNT column
            budget_df = clean_and_convert_amount(budget_df)
            if budget_df is None:
                return

            # Generate initial recommendations
            recommendations = generate_budget_recommendations(analysis_df, budget_df)

            # Convert budget values to billions
            budget_columns = ['AMOUNT', 'recommended_budget']
            recommendations[budget_columns] = recommendations[budget_columns] / 1e9

            # Sort recommendations from highest to lowest budget
            recommendations = recommendations.sort_values('AMOUNT', ascending=False).reset_index(drop=True)

            # Calculate percentage change
            recommendations['percent_change'] = (recommendations['recommended_budget'] - recommendations['AMOUNT']) / recommendations['AMOUNT'] * 100

            # Pagination
            items_per_page = 10
            num_pages = len(recommendations) // items_per_page + (1 if len(recommendations) % items_per_page > 0 else 0)
            page = st.selectbox("Select page", range(1, num_pages + 1))
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            st.subheader("Initial Recommendations")
            display_df = recommendations.iloc[start_idx:end_idx][['name', 'AMOUNT', 'recommended_budget', 'percent_change']].copy()
            display_df['AMOUNT'] = display_df['AMOUNT'].apply(format_budget)
            display_df['recommended_budget'] = display_df['recommended_budget'].apply(format_budget)
            display_df['percent_change'] = display_df['percent_change'].apply(lambda x: f"{x:.2f}%")
            st.table(display_df)

            # Allow user to adjust budgets
            st.subheader("Adjust Budgets")
            adjusted_budgets = {}
            for _, row in recommendations.iloc[start_idx:end_idx].iterrows():
                adjusted_budget = st.slider(
                    f"Adjust budget for {row['name']} (Billion Baht)",
                    min_value=float(row['AMOUNT'] * 0.8),
                    max_value=float(row['AMOUNT'] * 1.2),
                    value=float(row['recommended_budget']),
                    format="%.2f",
                    key=row['name']
                )
                adjusted_budgets[row['name']] = adjusted_budget

            # Update recommendations based on user adjustments
            if st.button("Update Recommendations"):
                # Create a new dataframe with user adjustments
                user_adjusted_df = pd.DataFrame(list(adjusted_budgets.items()), columns=['name', 'user_adjusted_budget'])
                merged_df = pd.merge(recommendations, user_adjusted_df, on='name', how='left')
                merged_df['adjustment'] = (merged_df['user_adjusted_budget'] - merged_df['AMOUNT']) / merged_df['AMOUNT']

                # Generate new recommendations
                new_recommendations = generate_budget_recommendations(analysis_df, budget_df)
                new_recommendations[budget_columns] = new_recommendations[budget_columns] / 1e9
                new_recommendations['user_adjusted_budget'] = merged_df['user_adjusted_budget']
                new_recommendations['percent_change'] = (new_recommendations['recommended_budget'] - new_recommendations['AMOUNT']) / new_recommendations['AMOUNT'] * 100
                new_recommendations['user_percent_change'] = (new_recommendations['user_adjusted_budget'] - new_recommendations['AMOUNT']) / new_recommendations['AMOUNT'] * 100

                st.subheader("Updated Recommendations")
                display_df = new_recommendations.iloc[start_idx:end_idx][['name', 'AMOUNT', 'recommended_budget', 'user_adjusted_budget', 'percent_change', 'user_percent_change']].copy()
                display_df['AMOUNT'] = display_df['AMOUNT'].apply(format_budget)
                display_df['recommended_budget'] = display_df['recommended_budget'].apply(format_budget)
                display_df['user_adjusted_budget'] = display_df['user_adjusted_budget'].apply(format_budget)
                display_df['percent_change'] = display_df['percent_change'].apply(lambda x: f"{x:.2f}%")
                display_df['user_percent_change'] = display_df['user_percent_change'].apply(lambda x: f"{x:.2f}%")
                st.table(display_df)

                # Visualizations
                st.subheader("Budget Visualizations")

                # 1. Budget Comparison Bar Chart
                fig_comparison = px.bar(new_recommendations.iloc[start_idx:end_idx], 
                                        x='name', 
                                        y=['AMOUNT', 'recommended_budget', 'user_adjusted_budget'],
                                        title="Budget Comparison",
                                        labels={'value': 'Budget (Billion Baht)', 'variable': 'Budget Type'},
                                        barmode='group')
                st.plotly_chart(fig_comparison, use_container_width=True)

                # 2. Percentage Change Comparison
                fig_percent = px.bar(new_recommendations.iloc[start_idx:end_idx], 
                                     x='name', 
                                     y=['percent_change', 'user_percent_change'],
                                     title="Percentage Change Comparison",
                                     labels={'value': 'Percentage Change', 'variable': 'Change Type'},
                                     barmode='group')
                st.plotly_chart(fig_percent, use_container_width=True)

                # 3. Budget Distribution Pie Chart
                fig_pie = px.pie(new_recommendations, values='user_adjusted_budget', names='name',
                                 title="Budget Distribution (User Adjusted)")
                st.plotly_chart(fig_pie, use_container_width=True)

                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input data and try again.")

if __name__ == "__main__":
    main()
