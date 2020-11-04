from tkinter import *
from tkinter import filedialog, messagebox
import tkinter.font as font
from PIL import ImageTk, Image
from pickle import load
from utils import *
tokenizer = load(open('tokenizer.pkl', 'rb'))

# -------------------------------------------

def centerWindow(branch, width, height):
    window_width = width
    window_height = height
    screen_width = branch.winfo_screenwidth()
    screen_height = branch.winfo_screenheight()
    x_cor = int((screen_width/2) - (window_width/2))
    y_cor = int((screen_height/3) - (window_height/3))
    branch.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cor, y_cor))

    
def switchButtonState1():
    if imageButton['state'] == DISABLED:
        imageButton['state'] = NORMAL
        imageButton['relief'] = RAISED
        
        
def switchButtonState2(button):
    if caption != '':
        button['state'] = DISABLED
        button['relief'] = SUNKEN
        
        
def load_weights():
    global model
    weight_path  = filedialog.askopenfilename(initialdir='/', title='Select file', filetypes=[('Model Weights', '*.h5' )])
    model = load_model(weight_path)
    response = messagebox.showinfo(title="Information", message='{} weights loaded'.format(weight_path.split('/')[-1]))
    
    
def display_prediction(window, caption):
    window.configure(state=NORMAL)
    window.insert(0,caption)
    window.configure(state='readonly')
    
    
def picture_button():
    global image_path
    
    image_path  = filedialog.askopenfilename(initialdir='/', title='Select image', filetypes=[("jpeg files","*.jpg"),("all files","*.*")])
    if image_path == '':
        messagebox.showinfo(title="Information", message="You have not selected a photo.")
        return
    
    top = Toplevel(root, background='white')
    centerWindow(top, width=450, height=500)
    top.title('Image')
    top.iconbitmap('icon.ico')
    top.resizable(0,0)
    top.grab_set()
    
    top_top = Frame(top, width=450, height=450, bg='white')
    top_top.pack()
    
    top_bot = Frame(top, width=450, height=50, bg='white')
    top_bot.pack()
          
    canvas2 = Canvas(top_top, bg='white', width=400, heigh=400, borderwidth=1, highlightthickness=5)
    canvas2.grid()
    my_image = Image.open(image_path)
    canvas2.my_image = ImageTk.PhotoImage(my_image.resize((400,400)))
    canvas2.create_image(206,206, image=canvas2.my_image)
    
    
    textWindow = Entry(top_bot, 
                       width=68, 
                       borderwidth=1, 
                       state=DISABLED)
    textWindow.grid(row=0, column=0, columnspan=2, sticky=N, pady=(10,5))
        
    predictButton = Button(top_bot, 
                           text='Predict', 
                           command = lambda: [get_caption(), display_prediction(textWindow, caption), switchButtonState2(predictButton)],
                           bg='#8ed8cb',
                           fg='white',
                           activebackground='#8ed8cb',
                           activeforeground='white')
    predictButton.grid(row=1, column=0, padx=(100,0), pady=(10,0), ipadx=15)
    
    backButton = Button(top_bot, 
                        text='Back', 
                        command= top.destroy, 
                        bg='#8ed8cb',
                        fg='white',
                        activebackground='#8ed8cb',
                        activeforeground='white')
    backButton.grid(row=1, column=1, padx=(0,100), pady=(10,0), ipadx=22)
        
        
def get_caption(): 
    global caption
    feature = extract_image_feature(image_path)
    description = generate_description(model, tokenizer, feature, 34)
    caption = clean_summary(description)
    
def exit_app():
    message_box = messagebox.askquestion('Exit Application', 'Are you sure you want to exit the application', icon = 'warning')
    if message_box == 'yes':
        root.destroy()
    
    
#----------------------------------------------------------------------------------------------------
    
root = Tk()
centerWindow(root, width=400, height=200)
root.title('My Caption App')
root.iconbitmap('icon.ico')
root.resizable(0,0)

top_frame = Frame(root, width=400, height=150)
top_frame.pack()

bottom_frame = Frame(root, bg='white', width=400, height=50)
bottom_frame.pack()
    
canvas1 = Canvas(top_frame,bg='#8ed8cb', width=400, heigh=150, borderwidth=0, highlightthickness=0)
canvas1.grid()
logo = Image.open('logo.png')
canvas1.logo = ImageTk.PhotoImage(logo.resize((100,100)))
canvas1.create_image(200,75, image=canvas1.logo)


weightsButton = Button(bottom_frame, 
                       text='Load Weights',  
                       command= lambda: [load_weights(), switchButtonState1()], 
                       bg='#8ed8cb',
                       fg='white',
                       relief=RAISED,
                       activebackground='#8ed8cb',
                       activeforeground='white')
weightsButton.grid(row=0, column=0, padx=20, pady=(10,14), ipadx=6)


imageButton = Button(bottom_frame, 
                     text='Load Image',  
                     command=picture_button, 
                     bg='#8ed8cb',
                     fg='white',
                     disabledforeground='white',
                     state= DISABLED,
                     relief=SUNKEN,
                     activebackground='#8ed8cb',
                     activeforeground='white')
imageButton.grid(row=0, column=1, padx=20, pady=(10,14), ipadx=10)


exitButtton = Button(bottom_frame, 
                     text='Exit program', 
                     command=exit_app,
                     bg='#8ed8cb',
                     fg='white',
                     relief=RAISED,
                     activebackground='#8ed8cb',
                     activeforeground='white')
exitButtton.grid(row=0, column=2, padx=20, pady=(10,14), ipadx=7)


mainloop( )