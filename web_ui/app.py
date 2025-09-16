from flask import Flask, render_template, request, jsonify
from capsule_movie_core.pipelines.text2video_pipeline import Text2VideoPipeline
import os

app = Flask(__name__)
pipeline = Text2VideoPipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate_video():
    data = request.json
    prompt = data.get('prompt', '')
    
    try:
        # TODO: Implement actual video generation
        # For now, return a mock response
        return jsonify({
            'status': 'success',
            'message': f'Started generating video for prompt: {prompt}',
            'job_id': '12345'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/status/<job_id>')
def get_status(job_id):
    # TODO: Implement actual status checking
    return jsonify({
        'status': 'processing',
        'progress': 50
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)