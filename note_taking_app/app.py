from flask import Flask, render_template, request

app = Flask(__name__)

notes = []

@app.route('/', methods=["GET", "POST"])  # Allow both GET and POST methods
def index():
    if request.method == "POST":  # Check if the request method is POST
        note = request.form.get("note")  # Retrieve note from form data
        if note:  # Check if note is not empty
            notes.append(note)  # Append note to notes list
    return render_template("home.html", notes=notes)

if __name__ == '__main__':
    app.run(debug=True)