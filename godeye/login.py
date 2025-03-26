import os
import customtkinter as ctk
import tkinter.messagebox as tkmb
import json
from camera import startapplication


# Selecting GUI theme - dark, light , system (for system default)
ctk.set_appearance_mode("dark")

# Selecting color theme - blue, green, dark-blue
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.geometry("430x400")
app.title("God's Eye")

def tab():

	# tkmb.showinfo(title="Login Successful",message="You have logged in Successfully")
	new_window = ctk.CTkToplevel(app)
	new_window.title("New Window")
	new_window.geometry("350x150")
	# tkmb.showinfo(title="Login Successful",message="You have logged in Successfully")
	ctk.CTkLabel(new_window,text="hii the program will be completed soon ").pack()
	startapplication()
	


def login():
    username = user_entry.get()
    password = user_pass.get()
    filename = "credentials.json"
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
	    with open(filename, "w") as f:
	        json.dump({}, f)
	    tkmb.showwarning("Warning", "Credentials file created")
    else:
        with open(filename, "r") as f:
            try:
                credentials = json.load(f)
            except json.JSONDecodeError as e:
                tkmb.showerror("Error", "Failed to read credentials: " + str(e))
                return
        if username in credentials and credentials[username] == password:
        	tkmb.showinfo(title="Login Successful",message="You have logged in Successfully")
        	tab()
			# tkmb.showinfo(title="Login Successful",message="You have logged in Successfully")
			# new_window = ctk.CTkToplevel(app)
			# new_window.title("New Window")
			# new_window.geometry("350x150")
			# tkmb.showinfo(title="Login Successful",message="You have logged in Successfully")
			# ctk.CTkLabel(new_window,text="hii the program will be started soon ").pack()


        else:
            tkmb.showerror("Error", "Invalid username or password")

def save():
    username = user_entry.get()
    password = user_pass.get()
    filename = "credentials.json"
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        credentials = {}
    else:
        with open(filename, "r") as f:
            try:
                credentials = json.load(f)
            except json.JSONDecodeError as e:
                tkmb.showerror("Error", "Failed to read credentials: " + str(e))
                return
    # if not username or not password:
	#     tkmb.showerror("Error", "Username or password cannot be empty")
	#     return

    credentials[username] = password
    with open(filename, "w") as f:
        json.dump(credentials, f)
    tkmb.showinfo("Success", "Credentials saved successfully!")


def singup():
	new_window = ctk.CTkToplevel(app)

	new_window.title("Sing Up Window")

	new_window.geometry("300x300")

	label = ctk.CTkLabel(master=new_window,text='Enter your details')
	label.pack(pady=12,padx=10)


	user_entry= ctk.CTkEntry(master=new_window,placeholder_text="Username") 
	user_entry.pack(pady=12,padx=10)

	user_pass= ctk.CTkEntry(master=new_window,placeholder_text="Password",show="*")
	user_pass.pack(pady=12,padx=10)
	button = ctk.CTkButton(master=new_window,text='Save',command=save)
	button.pack(pady=12,padx=10)


label = ctk.CTkLabel(app,text=" Login Page ")

label.pack(pady=10)


frame = ctk.CTkFrame(master=app)
frame.pack(pady=20,padx=40,fill='both',expand=True)

label = ctk.CTkLabel(master=frame,text='Accident Detection System')
label.pack(pady=12,padx=10)


user_entry= ctk.CTkEntry(master=frame,placeholder_text="Username")
user_entry.pack(pady=12,padx=10)

user_pass= ctk.CTkEntry(master=frame,placeholder_text="Password",show="*")
user_pass.pack(pady=12,padx=10)


button = ctk.CTkButton(master=frame,text='Login',command=login)
button.pack(pady=12,padx=10)
button = ctk.CTkButton(master=frame,text='Sing UP',command=singup)
button.pack(pady=12,padx=10)

checkbox = ctk.CTkCheckBox(master=frame,text='Remember Me')
checkbox.pack(pady=12,padx=10)


app.mainloop()


