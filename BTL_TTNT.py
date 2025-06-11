import sys             # thư viện hệ thống: cũng cấp quyền truy cập đến 1 số biến và hàm tương tác
from PyQt6.QtWidgets import QApplication
import sys
from PyQt6.QtWidgets import ( QMainWindow, QWidget, QLabel,
                             QComboBox, QVBoxLayout,  QPushButton,
                             QGridLayout, )


# QMainWindow	    Cửa sổ chính của ứng dụng, có thể chứa menu, thanh công cụ...
# QWidget	        Widget cơ bản dùng để làm container hoặc widget con.
# QLabel	        Hiển thị văn bản hoặc hình ảnh.
# QComboBox	        Danh sách chọn (dropdown menu).
# QVBoxLayout	    Bố cục sắp xếp các widget theo chiều dọc.
# QPushButton	    Nút bấm tương tác.
# QGridLayout	    Bố cục dạng lưới: sắp xếp widget theo hàng và cột.
# Qt	            Chứa các hằng số (constant), ví dụ: Qt.AlignmentFlag.AlignCenter để canh giữa.

from PyQt6.QtCore import Qt

import numpy as np
import pandas as pd          # thư viện thao tác dạng bảng
from sklearn.preprocessing import LabelEncoder          # dùng để chuyển đổi nhãn từ dạng chuỗi sang số

class HealthPredictor:    # lớp dự đoán sức khỏe
    def __init__(self):     # hàm khởi tạo
        self.label_encoders = {}   # là một từ điển lưu các đối tượng LabelEncoder để mã hóa/giải mã nhãn dạng chuỗi thành số và ngược lại cho từng cột (thuộc tính).

    def convert_value_to_label(self, attribute, value):  # Hàm dùng để chuyển đổi giá trị số trở lại chuỗi ban đầu (giải mã) theo từng thuộc tính.

        if isinstance(value, str):   # nếu value là chuỗi thì không cần giải mã vì nó đã là chuỗi rồi, trả về luôn
            return value

        # kiểm tra xem thuộc tính có nằm trong từ điển label_encoders không
        if attribute in self.label_encoders:
            try: # nếu có thì dùng inverse_transform để chuyển số về chuỗi gốc ban đầu
                return self.label_encoders[attribute].inverse_transform([value])[0]
            # nếu thuộc tính có mã hóa nhãn thì dùng inverse_transform để lấy lại chuỗi gốc

            except ValueError: # nếu bị lỗi, trả về value như cũ, không chuyển
                return value
        return value # nếu thuộc tính không nằm trong label_encoders cũng trả về value

    # Hàm đọc dữ liệu từ file CSV sử dụng pandas
    def load_data(self, path= 'Data.csv'): # Hàm đọc file CSV, mặc định tên là Data.csv
        data = pd.read_csv(path) # đọc file csv bằng pandas, lưu vào biến data
        columns_to_process = [col for col in data.columns if col != 'target'] # tạo danh sách tên cột không bao gồm cột kết quả (target)
        for col in columns_to_process: # duyệt qua từng cột cần xử lí
            # Kiểm tra xem cột có phải toàn số nguyên không
            if not data[col].dtype.kind in 'iu':  # 'i' cho integer, 'u' cho unsigned integer
                print(f"Xử lý cột {col}") # in ra cột đang xử lí
                self.label_encoders[col] = LabelEncoder() # tạo label_encoders mới cho cột đó
                data[col] = self.label_encoders[col].fit_transform(data[col]) # mã hóa các giá trị chuỗi thành số nguyên bằng label encoder
        return data # trả về data sau khi xử lí xong

    @staticmethod # hàm tĩnh, không cần tạo self
    def load_full_attributes(data): # hàm nhận đầu vào là data (dữ liệu đã được đọc từ file csv xử lí trước đó)
        data = data.dropna() # xóa các dòng có giá trị thiếu để đảm bảo dữ liệu đầu vào đầy đủ
        attributes = list(data.columns) # lấy danh sách tất cả các cột trong bảng dữ liệu
        attributes.remove('target') # Loại bor cột target vì nó không phải đầu vào mà là kết quả
        return attributes # trả về danh sách các cột còn lại (chính là các thuộc tính để huấn luyện)

    @staticmethod
    def save_processed_data(data, output_path="processed_heart.csv"):
        # Hàm Lưu dữ liệu đã chuyển đổi, không kèm theo chỉ số dòng
        data.to_csv(output_path, index=False)
    def save_label_encoders(self, output_path="label_encoders.json"): # Hàm lưu toàn bộ các LabelEncoder vào một file JSON.
        mapping_info = {} # tạo 1 từ điển để chứa thông tin mã hóa
        for col, encoder in self.label_encoders.items(): # duyệt qua từng cột và encoder
            # Chuyển đổi numpy.int64 sang int Python thông thường để lưu file JSON
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

    # Hàm tính entropy của tập dữ liệu
    @staticmethod
    def  cal_entropy(data):
        labels = data['target'].values # lấy cột target
        # counts:  [ni], _ : [ai] với ni là số lượng của ai trong target
        # ở đây counts: [a, b] và _ : [0,1] tức là trong target có a số 0 và b số 1
        _, counts = np.unique(labels, return_counts=True)  # đếm số lần xuất hiện của 0, 1 trong cột target
        # tính xác suất của mỗi nhãn: p(1): tỉ lệ giá trị 1 xuất hiện
        probabilities = counts / len(labels) # lấy số laanf xuất hiện chia tổng mẫu

        # tính entropy: - sum(p(1) * log2(p(1)) + p(0) * log2(p(0)))
        return - np.sum(probabilities * np.log2(probabilities))  # 0.9995090461828582, công thức tính entropy

    # Hàm chia dữ liệu thành các tập con dựa trên giá trị của thuộc tính
    @staticmethod # Hàm chia dữ liệu thành các tập con dựa trên giá trị của thuộc tính
    def partition(data, attribute): # đầu vào là data và attribute: thuộc tính để phân đoạn
        partitions = {} # khởi tạo 1 từ điển rỗng: lưu trữ các phân đoạn dựa trên giá trị thuộc tính
        for index, row in data.iterrows(): # duyệt qua từng hàng trong cột thuộc tính đang xét, index là chỉ số của hàng, row là 1 Series (dạng key-value) đại diện cho giá trị trong hàng đó
            value = row[attribute] # lấy giá trị thuộc tính attribute
            if value not in partitions: # nếu giá trị của thuộc tính không là 1 khóa trong từ điển thì tạo 1 khóa mới trong từ điển
                partitions[value] = []
            partitions[value].append(row.to_dict()) # thêm vào phân đoạn tương ứng với giá trị của thuộc tính
        return partitions # đầu ra là 1 từ điển với các giá trị thuộc tính làm khóa và các phân đoạn dữ liệu tương ứng là giá trị

    # Hàm tính thuộc tính tốt nhất để chia dữ liệu
    def calculate_information_gain(self, data, attribute):
        current_entropy = self.cal_entropy(data)   # Tính toán entropy hiện tại của toàn bộ dữ liệu
        attribute_entropy = 0.0 # khởi tạo entropy của thuộc tính bằng 0
        partitions = self.partition(data, attribute)  # hàm để phân chia dữ liệu theo thuộc tính (cái hàm ở trên)
        for partition_data in partitions.values(): # duyệt qua từng phân đoạn d liệu
            partition_entropy = self.cal_entropy(pd.DataFrame(partition_data))
            attribute_entropy += (len(partition_data) / len(data)) * partition_entropy #tính entropy của 1 thộc tính

        return current_entropy - attribute_entropy

    def calculate_attr_importance(self, data, attributes): # hàm đánh giá mức độ quan trọng của từng thuộc tính
        importance_dict = {} # từ điển rỗng lưu giá trị information gain của từng thuộc tính
        for attribute in attributes:  # duyệt qua từng thuộc tính trong danh sách
            importance = self.calculate_information_gain(data, attribute) # tính information gain của từng thuộc tính
            importance_dict[attribute] = importance # lưu kết quả tính được vào từ điển

        return importance_dict

    def find_best_attribute(self, data, attributes): # hàm tìm ra thuộc tính có tầm quan trọng cao nhất
        attr_importance = self.calculate_attr_importance(data, attributes)
        best_attr = max(attr_importance, key=attr_importance.get)

        return best_attr

    def build_decision_tree(self, data, attributes, max_depth=5,
                            current_depth=0):  # Định nghĩa hàm dựng cây quyết định với giới hạn độ sâu
        if (current_depth >= max_depth or  # Nếu đã đạt độ sâu tối đa
                len(attributes) == 0 or  # Hoặc không còn thuộc tính nào để chia
                len(data) == 0 or  # Hoặc không còn bản ghi dữ liệu
                len(data['target'].unique()) == 1):  # Hoặc tất cả bản ghi đều có cùng nhãn (chỉ 1 lớp)
            return data['target'].mode()[0]  # Trả về nhãn xuất hiện nhiều nhất làm nhãn dự đoán (lá)

        best_attr = self.find_best_attribute(data,
                                             attributes)  # Tìm thuộc tính tốt nhất để chia dữ liệu (ID3 - độ lợi thông tin)
        sub_tree = {}  # Khởi tạo cây con lưu các nhánh con

        new_attributes = [f for f in attributes if
                          f != best_attr]  # Loại bỏ thuộc tính vừa dùng khỏi danh sách thuộc tính

        for value in sorted(data[best_attr].unique()):  # Duyệt qua từng giá trị có thể của thuộc tính tốt nhất
            sub_data = data[data[best_attr] == value].copy()  # Lọc ra tập con dữ liệu có giá trị đó

            if len(sub_data) > 0:  # Nếu tập con không rỗng
                label_value = self.convert_value_to_label(best_attr, value)  # Gán nhãn hiển thị cho giá trị
                sub_tree[label_value] = self.build_decision_tree(  # Đệ quy để xây cây con
                    sub_data,
                    new_attributes,  # Sử dụng danh sách thuộc tính còn lại
                    max_depth,
                    current_depth + 1  # Tăng độ sâu thêm 1
                )
            else:  # Nếu tập con rỗng
                label_value = self.convert_value_to_label(best_attr, value)  # Gán nhãn cho giá trị rỗng
                sub_tree[label_value] = data['target'].mode()[0]  # Gán nhãn phổ biến làm kết quả mặc định

        return {
            best_attr: sub_tree}  # Trả về node hiện tại dưới dạng dictionary: {tên_thuộc_tính: {giá_trị: cây_con, ...}}

    def generate_rules(self, tree, attributes, rule_values=None):  # Định nghĩa hàm sinh luật từ cây quyết định
        if rule_values is None:  # Nếu chưa có điều kiện nào
            rule_values = {attr: None for attr in attributes}  # Khởi tạo dict với tất cả thuộc tính = None

        rules = []  # Danh sách chứa các luật

        if not isinstance(tree, dict):  # Nếu node hiện tại là lá (giá trị dự đoán)
            return [
                (rule_values.copy(), "Bệnh tim" if tree == 1 else "Không bệnh tim")]  # Trả về luật với điều kiện → nhãn

        attribute = list(tree.keys())[0]  # Lấy thuộc tính tại node hiện tại (key đầu tiên)
        branches = tree[attribute]  # Lấy các nhánh con ứng với từng giá trị của thuộc tính

        for value, subtree in branches.items():  # Duyệt qua từng nhánh con
            new_rule_values = rule_values.copy()  # Sao chép lại điều kiện hiện tại
            new_rule_values[attribute] = value  # Gán giá trị của thuộc tính hiện tại

            rules.extend(self.generate_rules(  # Gọi đệ quy cho cây con
                subtree,
                attributes,
                new_rule_values
            ))

        return rules  # Trả về danh sách các luật suy ra từ cây

    @staticmethod
    def print_rules(rules_df, filename='result/rules.txt'):  # Hàm tĩnh in tập luật từ DataFrame vào file text
        with open(filename, 'w', encoding='utf-8') as f:  # Mở file để ghi, sử dụng mã hóa UTF-8
            f.write("\nTập luật từ cây quyết định:\n")  # Ghi tiêu đề
            f.write("=" * 100 + "\n")  # Ghi dòng phân cách

            for i, row in enumerate(rules_df.itertuples(), 1):  # Duyệt từng dòng (luật) trong DataFrame
                f.write(f"Luật {i}:\n")  # Ghi tiêu đề luật
                f.write("NẾU\n")  # Bắt đầu phần điều kiện

                active_conditions = [  # Lọc các điều kiện có giá trị không null
                    f"    {col} = {value}"
                    for col, value in rules_df.iloc[i - 1].items()
                    if pd.notna(value) and col != 'target'  # Bỏ qua cột target và giá trị null
                ]
                f.write("\n".join(active_conditions) + "\n")  # Ghi từng điều kiện
                f.write(f"THÌ {rules_df.iloc[i - 1]['target']}\n")  # Ghi kết luận của luật
                f.write("-" * 100 + "\n")  # Ghi dòng phân cách

            print(f"\nĐã ghi {len(rules_df)} luật vào file {filename}")  # In ra số lượng luật đã ghi

    def save_rules_to_csv(self, rules, filename='rules.csv'):  # Hàm lưu luật vào file CSV
        rules_data = []  # Danh sách chứa các dictionary từng luật
        for conditions, prediction in rules:  # Duyệt từng luật (điều kiện, nhãn)
            rule_dict = conditions.copy()  # Sao chép điều kiện
            for attr, value in rule_dict.items():  # Duyệt từng thuộc tính trong điều kiện
                if value is not None:  # Nếu có giá trị
                        rule_dict[attr] = self.convert_value_to_label(attr, value)  # Chuyển về dạng nhãn (label)
            rule_dict['target'] = prediction  # Thêm nhãn kết quả vào luật
            rules_data.append(rule_dict)  # Thêm vào danh sách

        rules_df = pd.DataFrame(rules_data)  # Tạo DataFrame từ danh sách luật
        rules_df.to_csv(filename, index=False)  # Lưu vào file CSV
        print(f"\nĐã lưu {len(rules)} luật vào file {filename}")  # In ra thông báo

    def load_rules_from_csv(self, filename='result/rules.csv'):  # Hàm tải luật từ file CSV
        try:
            rules_df = pd.read_csv(filename)  # Cố gắng đọc file CSV
        except FileNotFoundError:  # Nếu không tìm thấy file
            print("Không tìm thấy file rules. Đang tạo luật mới...")  # Thông báo
            data = self.load_data()  # Tải dữ liệu gốc
            attributes = self.load_full_attributes(data)  # Tải danh sách thuộc tính
            self.save_processed_data(data, 'result/process_heart.csv')  # Lưu dữ liệu đã xử lý
            self.save_label_encoders('result/label_encoders.json')  # Lưu bộ mã hóa nhãn
            tree = self.build_decision_tree(data, attributes)  # Xây dựng cây quyết định
            rules = self.generate_rules(tree, attributes)  # Sinh luật từ cây
            self.save_rules_to_csv(rules, filename)  # Lưu luật vào CSV
            rules_df = self.load_rules_from_csv(filename)  # Gọi lại chính hàm này để nạp luật mới lưu
            self.print_rules(rules_df)  # In luật ra file văn bản
        print(f"Đã đọc {len(rules_df)} luật từ file {filename}")  # In thông báo đọc thành công
        return rules_df  # Trả về DataFrame chứa các luật

    def diagnose(self, patient_data, rules_df):  # Hàm chẩn đoán bệnh tim dựa trên luật và dữ liệu bệnh nhân
        attributes = self.load_full_attributes(rules_df)  # Lấy danh sách đầy đủ thuộc tính
        patient_labels = {}  # Tạo dict chứa nhãn của từng thuộc tính bệnh nhân
        for attr, value in patient_data.items():  # Duyệt qua từng thuộc tính của bệnh nhân
            patient_labels[attr] = self.convert_value_to_label(attr, value)  # Chuyển giá trị sang nhãn

        for idx, rule in rules_df.iterrows():  # Duyệt từng luật
            match = True  # Biến kiểm tra luật có khớp không
            matching_conditions = []  # Danh sách lưu điều kiện phù hợp

            for attr in attributes:  # Duyệt từng thuộc tính
                rule_value = rule[attr]  # Lấy giá trị trong luật
                if pd.isna(rule_value):  # Nếu luật không có điều kiện cho thuộc tính này → bỏ qua
                    continue
                if patient_labels[attr] != rule_value:  # Nếu không khớp → không khớp toàn luật
                    match = False
                    break
                matching_conditions.append(f"{attr} = {rule_value}")  # Ghi lại điều kiện phù hợp

            if match:  # Nếu tất cả điều kiện đều khớp
                return {
                    'diagnosis': rule['target'],  # Trả về kết quả chẩn đoán
                    'rule_id': idx + 1,  # Số thứ tự của luật khớp
                    'matching_conditions': ' VÀ '.join(matching_conditions)  # Ghép các điều kiện thành câu
                }

        return {  # Nếu không có luật nào phù hợp
            'diagnosis': 'Không thể chẩn đoán',
            'rule_id': None,
            'matching_conditions': 'Không có luật phù hợp'
        }

    class HealthPredictorGUI(QMainWindow):  # Lớp giao diện chính kế thừa từ QMainWindow
        def __init__(self):  # Hàm khởi tạo
            super().__init__()  # Gọi constructor của QMainWindow
            self.predictor = HealthPredictor()  # Tạo đối tượng xử lý chẩn đoán
            self.rules_df = self.predictor.load_rules_from_csv()  # Tải tập luật từ file
            self.init_ui()  # Gọi hàm khởi tạo giao diện

        def init_ui(self):  # Hàm tạo giao diện người dùng
            self.setWindowTitle('Heart Disease Predictor')  # Tiêu đề cửa sổ
            self.setGeometry(100, 100, 800, 600)  # Kích thước và vị trí cửa sổ

            central_widget = QWidget()  # Tạo widget trung tâm
            self.setCentralWidget(central_widget)  # Gắn widget trung tâm vào cửa sổ
            main_layout = QVBoxLayout(central_widget)  # Tạo layout chính theo chiều dọc

            grid_layout = QGridLayout()  # Tạo layout dạng lưới để chứa các input

            self.inputs = {}  # Dictionary lưu trữ các ô nhập dữ liệu

            # Tạo các trường nhập liệu với danh sách tùy chọn
            self.create_input_field('Age', ['Adult', 'MidleAge', 'Old'], grid_layout, 0)
            self.create_input_field('sex', [0, 1], grid_layout, 1)
            self.create_input_field('cp', [0, 1, 2, 3], grid_layout, 2)
            self.create_input_field('trest', ['High', 'Low', 'Normal'], grid_layout, 3)
            self.create_input_field('chol', ['Extreme', 'High Risk', 'Normal'], grid_layout, 4)
            self.create_input_field('fbs', [0, 1], grid_layout, 5)
            self.create_input_field('restecg', [0, 1, 2], grid_layout, 6)
            self.create_input_field('thalach', ['High', 'Low', 'Normal'], grid_layout, 7)
            self.create_input_field('exang', [0, 1], grid_layout, 8)
            self.create_input_field('oldpeak', ['High', 'Low', 'Normal'], grid_layout, 9)
            self.create_input_field('slope', [0, 1, 2], grid_layout, 10)
            self.create_input_field('ca', [0, 1, 2, 3, 4], grid_layout, 11)
            self.create_input_field('thal', [0, 1, 2, 3], grid_layout, 12)

            main_layout.addLayout(grid_layout)  # Thêm layout lưới vào layout chính

            # Tạo nút "Predict"
            predict_button = QPushButton('Predict', self)
            predict_button.clicked.connect(self.predict)  # Gắn sự kiện click gọi hàm predict
            predict_button.setStyleSheet("""  # Tùy chỉnh giao diện nút
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
            main_layout.addWidget(predict_button)  # Thêm nút vào giao diện

            # Tạo nhãn hiển thị kết quả
            self.result_label = QLabel('')
            self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Căn giữa nội dung
            self.result_label.setStyleSheet("""  # Tùy chỉnh kiểu chữ và khoảng cách
                QLabel {
                    font-size: 16px;
                    padding: 10px;
                    margin-top: 10px;
                    width: 300px;
                }
            """)
            main_layout.addWidget(self.result_label)  # Thêm nhãn vào layout chính

        def create_input_field(self, name, options, layout, row):  # Hàm tạo dòng nhập liệu gồm label và combobox
            label = QLabel(f'{name}:')  # Tạo label cho tên thuộc tính
            combo = QComboBox()  # Tạo combobox chứa các lựa chọn
            combo.addItems([str(opt) for opt in options])  # Thêm các lựa chọn vào combo
            layout.addWidget(label, row, 0)  # Gắn label vào cột 0, dòng row
            layout.addWidget(combo, row, 1)  # Gắn combo vào cột 1, dòng row
            self.inputs[name] = combo  # Lưu combo vào dictionary để truy xuất sau

        def predict(self):  # Hàm xử lý khi nhấn nút Predict
            patient_data = {}  # Dictionary chứa dữ liệu bệnh nhân
            for attr, combo in self.inputs.items():  # Duyệt qua tất cả các input
                value = combo.currentText()  # Lấy giá trị đã chọn
                try:
                    value = int(value)  # Nếu là số, chuyển sang kiểu int
                except ValueError:
                    pass  # Nếu không phải số, giữ nguyên dạng chuỗi
                patient_data[attr] = value  # Ghi vào dict

            print(patient_data)  # In dữ liệu bệnh nhân để kiểm tra

            # Gọi hàm chẩn đoán
            result = self.predictor.diagnose(patient_data, self.rules_df)

            # Tạo văn bản kết quả chẩn đoán
            result_text = f"""
            Diagnosis: {result['diagnosis']}
            Rule ID: {result['rule_id']}
            Matching Conditions: {result['matching_conditions']}
            """

            # Cập nhật nội dung hiển thị kết quả
            self.result_label.setText(result_text)

            # Đổi màu văn bản tùy theo kết quả
            if result['diagnosis'] == 'Bệnh tim':
                self.result_label.setStyleSheet("QLabel { color: red; }")  # Màu đỏ nếu bị bệnh
            else:
                self.result_label.setStyleSheet("QLabel { color: green; }")  # Màu xanh nếu bình thường

    def main():  # Hàm chính khởi tạo và chạy ứng dụng GUI
        app = QApplication(sys.argv)  # Tạo một đối tượng QApplication để quản lý vòng lặp sự kiện
        app.setStyle(
            'Fusion')  # Thiết lập giao diện (style) là 'Fusion' – giao diện hiện đại, nhất quán trên các hệ điều hành
        window = HealthPredictorGUI()  # Tạo cửa sổ chính từ lớp giao diện HealthPredictorGUI
        window.show()  # Hiển thị cửa sổ chính
        sys.exit(app.exec())  # Bắt đầu vòng lặp sự kiện của ứng dụng, thoát an toàn khi đóng ứng dụng

    if __name__ == '__main__':  # Kiểm tra nếu file này được chạy trực tiếp (không phải import từ module khác)
        main()  # Gọi hàm main() để khởi động ứng dụng
