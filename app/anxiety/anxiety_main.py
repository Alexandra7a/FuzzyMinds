import numpy as np
import pandas as pd


def get_anxiety_severity(gad_score):
    if 0 <= gad_score <= 4:
        return 'None-minimal'
    elif 5 <= gad_score <= 9:
        return 'Mild'
    elif 10 <= gad_score <= 14:
        return 'Moderate'
    else:
        return 'Severe'


def main():
    anxiety_file_path = '../data/anxiety_data.csv'

    anxietyData = pd.read_csv(anxiety_file_path)

    anxietyData.fillna('Null', inplace=True)

    if 'anxiety_treatment' in anxietyData.columns:
        anxietyData.drop('anxiety_treatment', axis=1, inplace=True)

    boolean_values = [True, False]
    genders = ['Male', 'Female']

    # num_entries = 11217
    #
    # new_gad_scores = np.random.randint(0, 21, num_entries)
    # new_anxiety_severities = [get_anxiety_severity(score) for score in new_gad_scores]
    #
    # new_anxiety_data = pd.DataFrame({
    #     'id': range(anxietyData['id'].max() + 1, anxietyData['id'].max() + num_entries + 1),
    #     'age': np.random.randint(13, 20, num_entries),
    #     'gender': np.random.choice(genders, num_entries),
    #     'gad_score': new_gad_scores,  # Use the generated GAD scores
    #     'anxiety_severity': new_anxiety_severities,  # Use the corresponding severities
    #     'anxiousness': np.random.choice(boolean_values, num_entries),
    #     'anxiety_diagnosis': np.random.choice(boolean_values, num_entries),
    #     'anxiety_treatment': np.random.choice(boolean_values, num_entries),
    # })
    #
    # combined_anxiety_data = pd.concat([anxietyData, new_anxiety_data], ignore_index=True)

    for index, row in anxietyData.iterrows():
        gad_score = row['gad_score']

        if row['anxiousness'] == 'Null':
            anxietyData.at[index, 'anxiousness'] = False if gad_score < 10 else True

        if row['anxiety_diagnosis'] == 'Null':
            anxietyData.at[index, 'anxiety_diagnosis'] = np.random.choice(boolean_values)

        anxietyData.at[index, 'anxiety_severity'] = get_anxiety_severity(gad_score)

    output_anxiety_path = '../data/anxiety_data.csv'
    anxietyData.to_csv(output_anxiety_path, index=False)
    # combined_anxiety_data.to_csv(output_anxiety_path, index=False)


if __name__ == '__main__':
    main()
