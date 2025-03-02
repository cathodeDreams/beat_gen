from beat_gen.webapp import app

if __name__ == '__main__':
    print("Starting Beat Generator Web App")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    app.run(debug=True)