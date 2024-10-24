import numpy as np
import pandas as pd


def get_depression_severity(phq_score):
    if 0 <= phq_score <= 4:
        return 'None-minimal'
    elif 5 <= phq_score <= 9:
        return 'Mild'
    elif 10 <= phq_score <= 14:
        return 'Moderate'
    elif 15 <= phq_score <= 19:
        return 'Moderately severe'
    else:
        return 'Severe'


def main():
    depression_file_path = 'C:/Users/aivla/Desktop/Andra/College/Anul III/Semestrul I/Metode inteligente de rezolvare a problemelor reale/Laboratoare/FirstDemo/data/depression_data.csv'
    depressionData = pd.read_csv(depression_file_path)

    depressionData.fillna('Null', inplace=True)
    depressionData.to_csv(depression_file_path, index=False)

    if 'depression_treatment' in depressionData.columns:
        depressionData.drop('depression_treatment', axis=1, inplace=True)

    boolean_values = [True, False]
    genders = ['Male', 'Female']

    # num_entries = 11217
    #
    # new_phq_score = np.random.randint(0, 27, num_entries)
    # new_depression_severity = [get_depression_severity(score) for score in new_phq_score]
    #
    # new_depression_data = pd.DataFrame({
    #     'id': range(depressionData['id'].max() + 1, depressionData['id'].max() + num_entries + 1),
    #     'age': np.random.randint(13, 20, num_entries),
    #     'gender': np.random.choice(genders, num_entries),
    #     'phq_score': new_phq_score,
    #     'depression_severity': new_depression_severity,
    #     'depressiveness': np.random.choice(boolean_values, num_entries),
    #     'suicidal': np.random.choice(boolean_values, num_entries),
    #     'depression_diagnosis': np.random.choice(boolean_values, num_entries),
    #     'depression_treatment': np.random.choice(boolean_values, num_entries)
    # })
    #
    # combined_depression_data = pd.concat([depressionData, new_depression_data], ignore_index=True)

    for index, row in depressionData.iterrows():
        phq_score = row['phq_score']

        if row['depressiveness'] == 'Null':
            depressionData.at[index, 'depressiveness'] = False if phq_score < 10 else True

        if row['suicidal'] == 'Null':
            depressionData.at[index, 'suicidal'] = False if phq_score < 15 else True

        if row['depression_diagnosis'] == 'Null':
            depressionData.at[index, 'depression_diagnosis'] = np.random.choice(boolean_values)

        depressionData.at[index, 'depression_severity'] = get_depression_severity(phq_score)

    output_depression_path = 'C:/Users/aivla/Desktop/Andra/College/Anul III/Semestrul I/Metode inteligente de rezolvare a problemelor reale/Laboratoare/FirstDemo/data/depression_data.csv'
    depressionData.to_csv(output_depression_path, index=False)
    # combined_depression_data.to_csv(output_depression_path, index=False)


if __name__ == '__main__':
    main()


