from flask import Blueprint, request, jsonify
import logging
import os
import json
from models.report_generator import ReportGenerator

report_bp = Blueprint('report', __name__)

@report_bp.route('/daily_report', methods=['GET'])
def daily_report():
    date_str = request.args.get('date')

    if not date_str:
        return jsonify({"error": "缺少日期参数"}), 400

    try:
        logging.debug(f"Received request for date: {date_str}")
        report_generator = ReportGenerator()
        report_data = report_generator.generate_daily_report(date_str)
        logging.debug(f"Generated report data: {report_data}")
        return jsonify(report_data)

    except Exception as e:
        logging.error(f"Error generating report: {str(e)}")
        return jsonify({
            "error": str(e),
            "date": date_str
        }), 500


@report_bp.route('/alert_history', methods=['GET'])
def api_alert_history():
    """获取异常提醒历史记录"""
    # Assuming logs directory is in the root
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    history_file = os.path.join(BASE_DIR, "logs", "alert_history.json")
    
    if not os.path.exists(history_file):
        return jsonify([])
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        # 倒序返回，最新的在前面
        return jsonify(history[::-1])
    except Exception as e:
        return jsonify({"error": str(e)}), 500
