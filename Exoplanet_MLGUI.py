import sys
from tkinter import *

from ExoplanetMyDecTreeClass import MODEL_TYPES, MyDecisionTree

window = Tk()
window.title("Exoplanet Predictor")
window.configure(bg="#f0f4ff")   # Light blue/white background

decisionTree = MyDecisionTree()

# ---------- UI COLORS ----------
TITLE_BG = "#2b4f81"
TITLE_FG = "white"
FRAME_BG = "#dbe6ff"
LABEL_FG = "#203864"
BUTTON_BG = "#4a90e2"
ENTRY_BG = "white"
ENTRY_FG = "black"   # set explicitly, otherwise macOS dark mode draws white text on the white boxes
TEXT_BG = "white"

# macOS ignores button background colours, so white button text would be invisible there.
BUTTON_FG = TITLE_BG if sys.platform == "darwin" else "white"
BUTTON_STYLE = dict(fg=BUTTON_FG, bg=BUTTON_BG, activeforeground=BUTTON_FG,
                    highlightbackground=FRAME_BG, font=("Arial", 10, "bold"))
ENTRY_STYLE = dict(bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG)


def show(message):
    text.delete("1.0", "end")
    text.insert(END, message)


def format_probabilities(probabilities):
    return '\n'.join(f'  {name:<15}{p:6.1%}' for name, p in
                     sorted(probabilities.items(), key=lambda item: -item[1]))


def trainData():
    decisionTree.updateSeed(int(seedsVar.get()))
    decisionTree.updateModelType(modelVar.get())
    show(f'Training {modelVar.get()}...')
    window.update_idletasks()
    decisionTree.trainData()
    examples = ', '.join(decisionTree.exampleTestKois())
    show(f'Data Trained! ({modelVar.get()}, seed {seedsVar.get()})\n\n'
         f'KOIs from the test set to try below:\n{examples}')


def testData():
    if not decisionTree.isTrained():
        show('Please train the model first.')
        return
    result, accuracy, macro_f1 = decisionTree.testData()
    lines = [f'Test set: {decisionTree.testSize()} objects',
             'Rows = actual, columns = predicted',
             ' ' * 15 + ''.join(f'{c[:9]:>11}' for c in decisionTree.categories())]
    for name, row in zip(decisionTree.categories(), result):
        lines.append(f'{name:<15}' + ''.join(f'{n:11d}' for n in row))
    lines.append(f'\nAccuracy {accuracy:.1%}    Macro F1 {macro_f1:.3f}')
    show('\n'.join(lines))


def predictKoi():
    if not decisionTree.isTrained():
        show('Please train the model first.')
        return
    result = decisionTree.predictKoi(koiEntry.get())
    if result is None:
        show(f'No Kepler object called "{koiEntry.get()}" in the dataset.\n'
             'Names look like K00752.01')
        return
    # Fill in the six fields so the user can see the object's values.
    for entry, value in zip(fieldEntries, result['core_values']):
        entry.delete(0, END)
        entry.insert(END, '' if value != value else f'{value:g}')   # value != value means NaN
    predictionEntry.delete(0, END)
    predictionEntry.insert(END, result['prediction'])
    note = ('\n\nNote: this object was in the training set, so this is not a fair test.'
            if result['in_training_set'] else '')
    show(f'Full model ({decisionTree.modelType}, all features)\n'
         f'Predicted: {result["prediction"]}\n'
         f'Actual:    {result["actual"]}\n\n'
         f'{format_probabilities(result["probabilities"])}{note}')


def makeNewPrediction():
    if not decisionTree.isTrained():
        show('Please train the model first.')
        return
    try:
        values = [float(entry.get()) for entry in fieldEntries]
    except ValueError:
        show('Please enter valid numeric values for all fields.')
        return  # Stop the function if input is invalid

    prediction_Res, probabilities = decisionTree.makePrediction(*values)
    predictionEntry.delete(0, END)
    predictionEntry.insert(END, prediction_Res)
    show('6-value model (less accurate than looking up a KOI)\n'
         f'Predicted: {prediction_Res}\n\n{format_probabilities(probabilities)}')


# ---------- LAYOUT ----------
# pack/grid (no fixed window size) so the window always fits its contents.
Label(window, text="Exoplanet ML Predictor", fg=TITLE_FG, bg=TITLE_BG,
      font=("Arial", 18, "bold"), padx=10, pady=10).pack(fill=X, padx=10, pady=(10, 4))
Label(window, text="Choose a model and seed, then train:", fg=LABEL_FG, bg=window["bg"],
      font=("Arial", 11, "bold")).pack(anchor=W, padx=14)

frame = Frame(window, bg=FRAME_BG, padx=10, pady=8)
frame.pack(fill=BOTH, expand=True, padx=10, pady=(4, 10))
frame.columnconfigure(1, weight=1)


def make_label(row, label, **grid):
    Label(frame, text=label, fg=LABEL_FG, bg=FRAME_BG, font=("Arial", 10, "bold"),
          anchor=W).grid(row=row, column=0, sticky=W+E, pady=3, padx=(0, 8), **grid)


def make_menu(row, label, variable, options):
    make_label(row, label)
    menu = OptionMenu(frame, variable, *options)
    menu.config(bg=ENTRY_BG, fg=ENTRY_FG, activeforeground=ENTRY_FG, highlightbackground=FRAME_BG)
    menu["menu"].config(bg=ENTRY_BG, fg=ENTRY_FG)
    menu.grid(row=row, column=1, sticky=W+E, pady=3)


def make_heading(row, label):
    Label(frame, text=label, fg=LABEL_FG, bg=FRAME_BG, font=("Arial", 14, "bold")).grid(
        row=row, column=0, columnspan=2, pady=(10, 2))


def create_field(row, label):
    make_label(row, label)
    ent = Entry(frame, **ENTRY_STYLE)
    ent.grid(row=row, column=1, sticky=W+E, pady=3)
    return ent


modelVar = StringVar(value=MODEL_TYPES[0])
make_menu(0, "Model", modelVar, MODEL_TYPES)
seedsVar = StringVar(value="1")
make_menu(1, "Seed Value", seedsVar, ['1', '2', '3', '4', '5'])

Button(frame, text="Train Data", command=trainData, **BUTTON_STYLE).grid(
    row=2, column=0, sticky=W+E, pady=4, padx=(0, 4))
Button(frame, text="Test Data", command=testData, **BUTTON_STYLE).grid(
    row=2, column=1, sticky=W+E, pady=4)

# Output
text = Text(frame, height=10, width=50, bg=TEXT_BG, fg=ENTRY_FG, insertbackground=ENTRY_FG,
            font=("Courier", 11), highlightthickness=0, wrap=WORD)
text.grid(row=3, column=0, columnspan=2, sticky=W+E, pady=4)

# Look up a real Kepler object
make_heading(4, "Predict a Kepler Object")
koiEntry = create_field(5, "KOI name (e.g. K00752.01)")
Button(frame, text="Look Up and Predict", command=predictKoi, **BUTTON_STYLE).grid(
    row=6, column=0, columnspan=2, sticky=W+E, pady=4)

# Manual entry
make_heading(7, "Or Enter Six Values")
fieldEntries = [
    create_field(8, "Orbital Period (days)"),
    create_field(9, "Impact Parameter"),
    create_field(10, "Transit Duration (hours)"),
    create_field(11, "Transit Depth (ppm)"),
    create_field(12, "Planet Radius (Earth radii)"),
    create_field(13, "Temperature, Teq (K)"),
]

Button(frame, text="Make Prediction", command=makeNewPrediction, **BUTTON_STYLE).grid(
    row=14, column=0, sticky=W+E, pady=4, padx=(0, 4))
predictionEntry = Entry(frame, **ENTRY_STYLE)
predictionEntry.grid(row=14, column=1, sticky=W+E, pady=4)

window.resizable(False, False)
window.mainloop()
