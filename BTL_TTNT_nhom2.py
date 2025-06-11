import sys
from PyQt6.QtWidgets import QApplication
import sys
from PyQt6.QtWidgets import ( QMainWindow, QWidget, QLabel,
                             QComboBox, QVBoxLayout,  QPushButton,
                             QGridLayout, )
from PyQt6.QtCore import Qt

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class HealthPredictor:
    def __init__(self):
        self.label_encoders = {}

    def convert_value_to_label(self, attribute, value):
        if isinstance(value, str):
            return value


        if attribute in self.label_encoders:
            try:
                return self.label_encoders[attribute].inverse_transform([value])[0]
            except ValueError:

                return value
        return value

    def load_data(self, path="Data.csv"):
        data = pd.read_csv(path)
        columns_to_process = [col for col in data.columns if col != 'target']
        for col in columns_to_process:
            if not data[col].dtype.kind in 'iu':
                print(f"Xử lý cột {col}")
                self.label_encoders[col] = LabelEncoder()
                data[col] = self.label_encoders[col].fit_transform(data[col])
        return data

    @staticmethod
    def load_full_attributes(data):
        data = data.dropna()
        attributes = list(data.columns)
        attributes.remove('target')
        return attributes

    @staticmethod
    def save_processed_data(data, output_path="processed_data.csv"):
        # Lưu dữ liệu đã chuyển đổi
        data.to_csv(output_path, index=False)
    def save_label_encoders(self, output_path="lable_encoders.json"):
        mapping_info = {}
        for col, encoder in self.label_encoders.items():
            # Chuyển đổi numpy.int64 sang int Python thông thường
            mapping_info[col] = {
                str(key): int(value)
                for key, value in zip(
                    encoder.classes_,
                    encoder.transform(encoder.classes_)
                )
            }

        # Lưu mapping vào file JSON
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, indent=4, ensure_ascii=False)

    @staticmethod
    def  cal_entropy(data):
        labels = data['target'].values
        # counts:  [ni], _ : [ai] với ni là số lượng của ai trong target
        # ở đây counts: [a, b] và _ : [0,1] tức là trong target có a số 0 và b số 1
        _, counts = np.unique(labels, return_counts=True)
        # tính xác suất của mỗi nhãn: p(1): tỉ lệ giá trị 1 xuất hiện
        probabilities = counts / len(labels)

        # tính entropy: - sum(p(1) * log2(p(1)) + p(0) * log2(p(0)))
        return - np.sum(probabilities * np.log2(probabilities))  # 0.9995090461828582

    @staticmethod
    def partition(data, attribute): # phân vùng
        partitions = {}
        for index, row in data.iterrows():
            value = row[attribute] # lấy giá trị thuộc tính attribute
            if value not in partitions:
                partitions[value] = []
            partitions[value].append(row.to_dict())
        return partitions

    def calculate_information_gain(self, data, attribute):
        current_entropy = self.cal_entropy(data)
        attribute_entropy = 0.0
        partitions = self.partition(data, attribute)
        for partition_data in partitions.values():
            partition_entropy = self.cal_entropy(pd.DataFrame(partition_data))
            attribute_entropy += (len(partition_data) / len(data)) * partition_entropy

        return current_entropy - attribute_entropy

    def calculate_attr_importance(self, data, attributes):
        importance_dict = {}
        for attribute in attributes:
            importance = self.calculate_information_gain(data, attribute)
            importance_dict[attribute] = importance

        return importance_dict

    def find_best_attribute(self, data, attributes):
        attr_importance = self.calculate_attr_importance(data, attributes)
        best_attr = max(attr_importance, key=attr_importance.get)

        return best_attr

    def build_decision_tree(self, data, attributes, max_depth=5, current_depth=0):
        # Xây dựng cây quyết định với giới hạn độ sâu
        if (current_depth >= max_depth or  # đạt độ sâu tối đa
                len(attributes) == 0 or  # hết thuộc tính
                len(data) == 0 or  # hết dữ liệu
                len(data['target'].unique()) == 1):  # chỉ còn 1 nhãn
            return data['target'].mode()[0]
        best_attr = self.find_best_attribute(data, attributes)
        sub_tree = {}

        # Loại bỏ thuộc tính đã sử dụng khỏi danh sách
        new_attributes = [f for f in attributes if f != best_attr]

        # Xây dựng các nhánh con
        for value in sorted(data[best_attr].unique()):
            sub_data = data[data[best_attr] == value].copy()
            if len(sub_data) > 0:
                label_value = self.convert_value_to_label(best_attr, value)
                sub_tree[label_value] = self.build_decision_tree(
                    sub_data,
                    new_attributes,
                    max_depth,
                    current_depth + 1
                )
            else:
                label_value = self.convert_value_to_label(best_attr, value)
                sub_tree[label_value] = data['target'].mode()[0]

        return {best_attr: sub_tree}

    def generate_rules(self, tree, attributes,  rule_values=None):
        if rule_values is None:
            # Khởi tạo dictionary với tất cả thuộc tính là null
            rule_values = {attr: None for attr in attributes}

        rules = []

        # Nếu node là lá (giá trị dự đoán)
        if not isinstance(tree, dict):
            return [(rule_values.copy(), "Bệnh tim" if tree == 1 else "Không bệnh tim")]

        # Lấy thuộc tính gốc và các nhánh
        attribute = list(tree.keys())[0]
        branches = tree[attribute]

        # Duyệt qua từng nhánh
        for value, subtree in branches.items():
            # Cập nhật giá trị cho thuộc tính hiện tại
            new_rule_values = rule_values.copy()
            new_rule_values[attribute] = value

            # Đệ quy với cây con
            rules.extend(self.generate_rules(subtree, attributes, new_rule_values))

        return rules

    @staticmethod
    def print_rules(rules_df, filename='result/rules.txt'):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\nTập luật từ cây quyết định:\n")
            f.write("=" * 100 + "\n")

            for i, row in enumerate(rules_df.itertuples(), 1):
                f.write(f"Luật {i}:\n")
                f.write("NẾU\n")

                # Ghi các điều kiện không null
                active_conditions = [
                    f"    {col} = {value}"
                    for col, value in rules_df.iloc[i - 1].items()
                    if pd.notna(value) and col != 'target'
                ]
                f.write("\n".join(active_conditions) + "\n")
                f.write(f"THÌ {rules_df.iloc[i - 1]['target']}\n")
                f.write("-" * 100 + "\n")

            print(f"\nĐã ghi {len(rules_df)} luật vào file {filename}")

    def save_rules_to_csv(self, rules, filename='rules.csv'):
        """
        Lưu tập luật vào file CSV với đầy đủ các cột
        """
        # Tạo list các dictionary cho DataFrame
        rules_data = []
        for conditions, prediction in rules:
            rule_dict = conditions.copy()
            # Chuyển đổi giá trị số thành nhãn cho các thuộc tính
            for attr, value in rule_dict.items():
                if value is not None:
                    rule_dict[attr] = self.convert_value_to_label(attr, value)
            rule_dict['target'] = prediction
            rules_data.append(rule_dict)

        # Tạo DataFrame và lưu vào CSV
        rules_df = pd.DataFrame(rules_data)
        rules_df.to_csv(filename, index=False)
        print(f"\nĐã lưu {len(rules)} luật vào file {filename}")

    def load_rules_from_csv(self, filename='result/rules.csv'):
        try:
            rules_df = pd.read_csv(filename)
        except FileNotFoundError:
            print("Không tìm thấy file rules. Đang tạo luật mới...")
            # Load dữ liệu và tạo cây quyết định
            data = self.load_data()
            attributes = self.load_full_attributes(data)
            self.save_processed_data(data, 'result/process_data.csv')
            self.save_label_encoders('result/label_encoders.json')
            tree = self.build_decision_tree(data, attributes)
            rules = self.generate_rules(tree, attributes)
            self.save_rules_to_csv(rules, filename)
            rules_df = self.load_rules_from_csv(filename)
            self.print_rules(rules_df)
        print(f"Đã đọc {len(rules_df)} luật từ file {filename}")
        return rules_df
    def diagnose(self, patient_data, rules_df):
        attributes = self.load_full_attributes(rules_df)
        # Chuyển đổi giá trị số của bệnh nhân thành nhãn
        patient_labels = {}
        for attr, value in patient_data.items():
            patient_labels[attr] = self.convert_value_to_label(attr, value)

        # Kiểm tra từng luật
        for idx, rule in rules_df.iterrows():
            match = True
            matching_conditions = []

            # Kiểm tra từng thuộc tính trong luật
            for attr in attributes:
                rule_value = rule[attr]
                # Bỏ qua các thuộc tính null trong luật
                if pd.isna(rule_value):
                    continue

                # So sánh giá trị của bệnh nhân với luật
                if patient_labels[attr] != rule_value:
                    match = False
                    break
                matching_conditions.append(f"{attr} = {rule_value}")

            # Nếu tất cả điều kiện đều khớp
            if match:
                return {
                    'diagnosis': rule['target'],
                    'rule_id': idx + 1,
                    'matching_conditions': ' VÀ '.join(matching_conditions)
                }

        return {
            'diagnosis': 'Không thể chẩn đoán',
            'rule_id': None,
            'matching_conditions': 'Không có luật phù hợp'
        }

class HealthPredictorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.predictor = HealthPredictor()
        self.rules_df = self.predictor.load_rules_from_csv()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Heart Disease Predictor')
        self.setGeometry(100, 100, 800, 600)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create grid layout for input fields
        grid_layout = QGridLayout()

        # Define input fields
        self.inputs = {}

        # Age options
        age_options = ['Adult', 'MidleAge', 'Old']
        self.create_input_field('Age', age_options, grid_layout, 0)

        # Sex options
        sex_options = [0, 1]
        self.create_input_field('sex', sex_options, grid_layout, 1)

        # Chest Pain Type options
        cp_options = [0, 1, 2, 3]
        self.create_input_field('cp', cp_options, grid_layout, 2)

        # Resting Blood Pressure options
        trest_options = ['High','Low','Normal']
        self.create_input_field('trest', trest_options, grid_layout, 3)

        # Cholesterol options
        chol_options = ['Extreme', 'High Risk', 'Normal']
        self.create_input_field('chol', chol_options, grid_layout, 4)

        # Fasting Blood Sugar options
        fbs_options = [0, 1]
        self.create_input_field('fbs', fbs_options, grid_layout, 5)

        # Resting ECG options
        restecg_options = [0, 1, 2]
        self.create_input_field('restecg', restecg_options, grid_layout, 6)

        # Maximum Heart Rate options
        thalach_options = ['High','Low','Normal']
        self.create_input_field('thalach', thalach_options, grid_layout, 7)

        # Exercise Induced Angina options
        exang_options = [0, 1]
        self.create_input_field('exang', exang_options, grid_layout, 8)

        # ST Depression options
        oldpeak_options = ['High','Low','Normal']
        self.create_input_field('oldpeak', oldpeak_options, grid_layout, 9)

        # Slope options
        slope_options = [0, 1, 2]
        self.create_input_field('slope', slope_options, grid_layout, 10)

        # Number of Major Vessels options
        ca_options = [0, 1, 2, 3, 4]
        self.create_input_field('ca', ca_options, grid_layout, 11)

        # Thalassemia options
        thal_options = [0, 1, 2, 3]
        self.create_input_field('thal', thal_options, grid_layout, 12)

        main_layout.addLayout(grid_layout)

        # Create predict button
        predict_button = QPushButton('Predict', self)
        predict_button.clicked.connect(self.predict)
        predict_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        main_layout.addWidget(predict_button)

        # Create result label
        self.result_label = QLabel('')
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                padding: 10px;
                margin-top: 10px;
                width: 300px;
            }
        """)
        main_layout.addWidget(self.result_label)

    def create_input_field(self, name, options, layout, row):
        label = QLabel(f'{name}:')
        combo = QComboBox()
        combo.addItems([str(opt) for opt in options])
        layout.addWidget(label, row, 0)
        layout.addWidget(combo, row, 1)
        self.inputs[name] = combo

    def predict(self):
        # Get values from input fields and convert to appropriate format
        patient_data = {}
        for attr, combo in self.inputs.items():
            value = combo.currentText()
            try:
                value = int(value)
            except ValueError:
                pass
            patient_data[attr] = value
        print(patient_data)
        # Get prediction
        result = self.predictor.diagnose(patient_data, self.rules_df)

        # Display result
        result_text = f"""
        Diagnosis: {result['diagnosis']}
        Rule ID: {result['rule_id']}
        Matching Conditions: {result['matching_conditions']}
        """


        # Update result label
        self.result_label.setText(result_text)
        if result['diagnosis'] == 'Bệnh tim':
            self.result_label.setStyleSheet("QLabel { color: red; }")
        else:
            self.result_label.setStyleSheet("QLabel { color: green; }")
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for a modern look
    window = HealthPredictorGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()