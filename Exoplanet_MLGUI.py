from tkinter import *

window = Tk()
window.geometry("600x750")
window.title("Exoplanet Predictor")
window.configure(bg="#f0f4ff")   # Light blue/white background


from ExoplanetMyDecTreeClass import MyDecisionTree

decisionTree = MyDecisionTree()

def trainData():
    newSeedValue = int(seedsVar.get())
    decisionTree.updateSeed(newSeedValue)
    decisionTree.trainData()
    text.delete("1.0","end")
    text.insert(END, 'Data Trained!\n')

def testData():
    text.delete("1.0","end")
    if not decisionTree.isTrained():
        text.insert(END, 'Please train the model first.')
        return
    text.insert(END, 'Rows = actual, columns = predicted\n')
    text.insert(END, decisionTree.readCategories() + '\n')
    result, accuracy = decisionTree.testData()
    text.insert(END, '\n'.join(' '.join(f'{n:4d}' for n in row) for row in result))
    text.insert(END, f'\n\nAccuracy {accuracy:.1%}')

def makeNewPrediction():
    if not decisionTree.isTrained():
        text.delete("1.0","end")
        text.insert(END, 'Please train the model first.')
        return
    try:
        period = float(entry41.get())
        impact = float(entry42.get())
        duration = float(entry43.get())
        depth = float(entry44.get())
        prad = float(entry45.get())
        teq = float(entry46.get())

    except ValueError:
        text.delete("1.0","end")
        text.insert(END, 'Please enter valid numeric values for all fields.')
        return  # Stop the function if input is invalid

    prediction_Res = decisionTree.makePrediction(period, impact, duration, depth, prad, teq)
    entry47.delete(0, END)
    entry47.insert(END, prediction_Res)

# ---------- UI COLORS ----------
TITLE_BG = "#2b4f81"
TITLE_FG = "white"
FRAME_BG = "#dbe6ff"
LABEL_FG = "#203864"
BUTTON_BG = "#4a90e2"
BUTTON_FG = "white"
ENTRY_BG = "white"
TEXT_BG = "white"

frame = Frame(window, width=550, height=350, bg=FRAME_BG)
frame.place(x=10, y=100)

label0 = Label(window, text="Exoplanet ML Predictor",
               fg=TITLE_FG, bg=TITLE_BG, font=("Arial", 18, "bold"), padx=10, pady=10)
label0.place(x=55, y=25)

label01 = Label(window,
                text="Enter seed value and train the model:",
                fg=LABEL_FG, bg=window["bg"], font=("Arial", 11, "bold"))
label01.place(x=30, y=80)

# Seed
label1 = Label(frame, text="Seed Value",
               fg=LABEL_FG, bg=FRAME_BG, width=15, font=("Arial", 10, "bold"))
label1.grid(row=0, column=0, sticky=W+E, pady=4)

list1 = ['1', '2', '3', '4', '5']
seedsVar = StringVar()
combo1 = OptionMenu(frame, seedsVar, *list1)
seedsVar.set("1")
combo1.config(bg="white")
combo1.grid(row=0, column=1, sticky=W+E, pady=4)

# Buttons
button1 = Button(frame, text="Train Data", fg=BUTTON_FG, bg=BUTTON_BG,
                 font=("Arial", 10, "bold"), command=trainData)
button1.grid(row=1, column=0, sticky=W+E, pady=4)

button2 = Button(frame, text="Test Data", fg=BUTTON_FG, bg=BUTTON_BG,
                 font=("Arial", 10, "bold"), command=testData)
button2.grid(row=1, column=1, sticky=W+E, pady=4)

# Output
label3 = Label(frame, text="Output", fg=LABEL_FG, bg=FRAME_BG,
               width=15, font=("Arial", 10, "bold"))
label3.grid(row=2, column=0)

text = Text(frame, height=8, width=7, bg=TEXT_BG)
text.grid(row=2, column=1, ipadx=100)

# Prediction Section
label03 = Label(frame, text="Predict New Planet Data",
                fg=LABEL_FG, bg=FRAME_BG, font=("Arial", 14, "bold"))
label03.grid(row=3, column=0, columnspan=2, pady=10)

def create_field(row, label):
    lbl = Label(frame, text=label, fg=LABEL_FG, bg=FRAME_BG,
                width=18, font=("Arial", 10, "bold"))
    lbl.grid(row=row, column=0, sticky=W+E, pady=3)
    ent = Entry(frame, bg=ENTRY_BG)
    ent.grid(row=row, column=1, sticky=W+E, pady=3)
    return ent

entry41 = create_field(4, "Orbital Period")
entry42 = create_field(5, "Impact Parameter")
entry43 = create_field(6, "Transit Duration")
entry44 = create_field(7, "Transit Depth")
entry45 = create_field(8, "Planetary Radius")
entry46 = create_field(9, "Temp (Teq)")

button3 = Button(frame, text="Make Prediction", fg=BUTTON_FG, bg=BUTTON_BG,
                 font=("Arial", 10, "bold"), command=makeNewPrediction)
button3.grid(row=10, column=0, sticky=W+E, pady=5)

entry47 = Entry(frame, bg=ENTRY_BG)
entry47.grid(row=10, column=1, sticky=W+E, pady=5)

window.mainloop()
