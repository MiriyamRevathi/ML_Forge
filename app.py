"""
MLForge - Production Machine Learning Systems Platform
Application Entry Point & Flask App Factory
"""

import sys
import logging
from flask import Flask, render_template, jsonify
from config import SECRET_KEY, MAX_CONTENT_LENGTH, BASE_DIR

# Import blueprints
from routes.dashboard import dashboard_bp
from routes.datasets import datasets_bp
from routes.quality import quality_bp
from routes.pipelines import pipelines_bp
from routes.experiments import experiments_bp
from routes.models import models_bp
from routes.predictions import predictions_bp
from routes.monitoring import monitoring_bp
from routes.reports import reports_bp
from routes.diagnostics import diagnostics_bp
from routes.settings import settings_bp


def create_app() -> Flask:
    """
    Application Factory creating and configuring the Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )

    # App Configurations
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    # Register Blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(datasets_bp)
    app.register_blueprint(quality_bp)
    app.register_blueprint(pipelines_bp)
    app.register_blueprint(experiments_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(predictions_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(reports_bp)
    app.register_blueprint(diagnostics_bp)
    app.register_blueprint(settings_bp)

    # Custom Error Handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template(
            "base.html",
            error_title="404 - Page Not Found",
            error_message="The requested page or resource could not be found on MLForge platform."
        ), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template(
            "base.html",
            error_title="500 - Internal Platform Error",
            error_message="An unexpected system error occurred while executing the request. Please check the logs."
        ), 500

    @app.context_processor
    def inject_global_variables():
        return {
            "platform_name": "MLForge",
            "version": "1.0.0"
        }

    return app


app = create_app()

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Launching MLForge — Machine Learning Systems Platform")
    print("   Server running locally at: http://127.0.0.1:5000")
    print("   System Diagnostics available at: http://127.0.0.1:5000/diagnostics")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
