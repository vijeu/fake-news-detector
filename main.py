import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import tkinter.messagebox as messagebox

# ملزومات برای اجرای درست برنامه
'''download stopwords if not already downloaded
nltk.download('stopwords')'''

# مسیرها
dataset_path = '/dataset'
fake_csv_path = os.path.join(dataset_path, 'Fake.csv')
true_csv_path = os.path.join(dataset_path, 'True.csv')
model_path = os.path.join(dataset_path, 'model.pkl')

# تعریف متغییرها
classifier = None
countv = None
f = None
t = None
X_train = None
X_test = None
y_train = None
y_test = None

# ایجاد صفحه‌ی اصلی برنامه
window = ctk.CTk()
window.title("تشخیص اخبار جعلی")
window.geometry("400x400")

def train_load_algorithms():
    global classifier, countv, f, t, X_train, X_test, y_train, y_test
    
    # بارگیری مجموعه‌ی داده
    f = pd.read_csv(fake_csv_path, delimiter=',')
    t = pd.read_csv(true_csv_path, delimiter=',')
    
    # برای محتوای جعلی 0 و برای واقعی 1
    f['temp'] = 0
    t['temp'] = 1
    
    # الحاق مجموعه‌ی داده
    datas = pd.concat([t, f], ignore_index=True)
    column = ['date', 'subject']
    datas = datas.drop(columns=column)
    input_arr = np.array(datas['title'])
    
    # پیش پردازش
    corpus = []
    
    for i in range(0, len(input_arr)):
        newArr = re.sub('[^a-zA-Z]', ' ', input_arr[i])
        newArr = newArr.lower()
        newArr = newArr.split()
        ps = PorterStemmer()
        newArr = [ps.stem(word) for word in newArr if not word in set(stopwords.words('english'))]
        newArr = ' '.join(newArr)
        corpus.append(newArr)
    
    countv = CountVectorizer(max_features=5000)
    X = countv.fit_transform(corpus).toarray()
    y = datas.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)
    
    joblib.dump((classifier, countv), model_path)
    
    train_load_button.config(state=tk.DISABLED)
    select_files_button.config(state=tk.NORMAL)

def open_files():
    files = filedialog.askopenfiles(filetypes=[('Text Files', '*.txt')])
    if files:
        result_listbox.delete(0, tk.END)  # حذف محتوای لیست-باکس قیل از نمایش نتایج
        for file in files:
            process_file(file.name)
    else:
        messagebox.showwarning('هیچ خبری انتخاب نشد', '.لطفاً حداقل یک خبر را انتخاب کنید')

def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    input_arr = [content]
    corpus = []

    for i in range(0, len(input_arr)):
        newArr = re.sub('[^a-zA-Z]', ' ', input_arr[i])
        newArr = newArr.lower()
        newArr = newArr.split()
        ps = PorterStemmer()
        newArr = [ps.stem(word) for word in newArr if not word in set(stopwords.words('english'))]
        newArr = ' '.join(newArr)
        corpus.append(newArr)

    X_test = countv.transform(corpus).toarray()
    y_pred = classifier.predict(X_test)
    prediction = "جعلی" if y_pred[0] == 0 else "واقعی"
    file_name = os.path.basename(file_path)
    result_listbox.insert(tk.END, f".\"{prediction}\n\n\" پیش‌‌بینی برابر است با [{file_name}\n]:برای خبر")

def import_model():
    global classifier, countv
    
    model_file = filedialog.askopenfilename(filetypes=[('Model Files', '*.pkl')])
    if model_file:
        classifier, countv = joblib.load(model_file)
        train_load_button.configure(state=tk.DISABLED)
        select_files_button.configure(state=tk.NORMAL)

# دکمه ها
train_load_button = ctk.CTkButton(window, text="آموزش دادن ماشین", font=('Calibri' , 16), command=train_load_algorithms)
train_load_button.pack(pady=(10,0))

import_model_button = ctk.CTkButton(window, text="وارد کردن مدل", font=('Calibri' , 15), command=import_model)
import_model_button.pack(pady=(10,0))

select_files_button = ctk.CTkButton(window, text="انتخاب اخبار", font=('Calibri' , 15), command=open_files, state=tk.DISABLED)
select_files_button.pack(pady=(30,20))

# ساخت قاب برای لیست-باکس و نوار پیمایش
listbox_frame = tk.Frame(window)
listbox_frame.pack(pady=0)

# لیست-باکس
result_listbox = tk.Listbox(listbox_frame, width=50 , font='Calibri' , justify='right')
result_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# نوار پیمایش عمودی
vertical_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=result_listbox.yview)
vertical_scrollbar.place(relx=0.017 , rely=1 , relheight=1 , anchor=tk.S)

# نوار پیمایش افقی
horizontal_scrollbar = tk.Scrollbar(listbox_frame, orient=tk.HORIZONTAL, command=result_listbox.xview)
horizontal_scrollbar.place(relx=0.52, rely=1, relwidth=0.96, anchor=tk.S)

# تنظیم
result_listbox.config(yscrollcommand=vertical_scrollbar.set, xscrollcommand=horizontal_scrollbar.set)
result_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# تنظیم کردن موقعیت پنجره در هنگام بازشدن
root_width = 440
root_height = 370
display_width = window.winfo_screenwidth()
display_height = window.winfo_screenheight()
left = int(display_width / 2 - root_width / 2)
top = int(display_height / 2 - root_height / 2)
window.geometry(f'{root_width}x{root_height}+{left}+{top}')
window.resizable(False, False)

# وارد کردن آرم برای برنامه و اجرای برنامه
window.iconbitmap('/icon/icon.ico')
window.mainloop()