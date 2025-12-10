from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, make_response, session
import csv
import sys
import os
import logging
import chardet
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import io
import math
import pandas as pd
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pickle
from threading import Thread

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import DataCleaner, DataAnalytics, ReportGenerator, SupabaseClient

app = Flask(__name__)
app.config['DATA_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SECRET_KEY'] = 'secretkey'

# Initialize Supabase client
db = SupabaseClient()

# Create data folder if it doesn't exist
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Cache directory for storing dataset metadata
CACHE_DIR = 'dataset_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Helper function for progress tracking
def set_upload_progress(dataset_id, current, total, stage):
    """Store upload progress in session"""
    session[f'upload_progress_{dataset_id}'] = {
        'current': current,
        'total': total,
        'stage': stage,
        'percentage': (current / total * 100) if total > 0 else 0
    }
    session.modified = True

def get_upload_progress(dataset_id):
    """Get upload progress from session"""
    return session.get(f'upload_progress_{dataset_id}')

# Helper functions for dataset management
def get_dataset_list():
    """Get list of all dataset IDs from session"""
    return session.get('dataset_list', [])

def set_dataset_list(dataset_list):
    """Save dataset list to session"""
    session['dataset_list'] = dataset_list

def get_current_dataset_id():
    """Get current dataset ID from session"""
    return session.get('current_dataset_id')

def set_current_dataset_id(dataset_id):
    """Set current dataset ID in session"""
    session['current_dataset_id'] = dataset_id

def save_dataset_metadata(dataset_id, metadata):
    """Save dataset metadata to file"""
    filepath = os.path.join(CACHE_DIR, f'{dataset_id}_meta.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(metadata, f)

def load_dataset_metadata(dataset_id):
    """Load dataset metadata from file"""
    filepath = os.path.join(CACHE_DIR, f'{dataset_id}_meta.pkl')
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def get_all_datasets():
    """Get all datasets metadata as a dictionary"""
    dataset_list = get_dataset_list()
    datasets = {}
    for dataset_id in dataset_list:
        metadata = load_dataset_metadata(dataset_id)
        if metadata:
            datasets[dataset_id] = metadata
    return datasets

def get_current_dataset():
    """Get current dataset metadata"""
    dataset_id = get_current_dataset_id()
    if dataset_id:
        return load_dataset_metadata(dataset_id)
    return None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return "File is too large. Maximum allowed size is 50MB.", 413

@app.route('/')
def index():
    dataset_id = request.args.get('dataset_id', get_current_dataset_id())
    if dataset_id:
        set_current_dataset_id(dataset_id)
    
    datasets = get_all_datasets()
    dataset = get_current_dataset()
    
    if not dataset:
        return render_template('index.html',
                             data=[],
                             datasets=datasets,
                             current_dataset_id=get_current_dataset_id())
    
    page = request.args.get('page', 1, type=int)
    included_page = request.args.get('included_page', 1, type=int)
    excluded_page = request.args.get('excluded_page', 1, type=int)
    per_page = 50
    
    # Get filters
    name_filter = request.args.get('name_filter', '').strip()
    month_filter = request.args.get('month_filter', '').strip()
    year_filter = request.args.get('year_filter', '').strip()
    day_filter = request.args.get('day_filter', '').strip()
    sort_by = request.args.get('sort_by', '')
    sort_order = request.args.get('sort_order', 'asc')
    
    # Fetch data from Supabase
    table_name = dataset['table_name']
    included_table = dataset['included_table']
    excluded_table = dataset['excluded_table']
    
    # Fetch original data
    csv_data = db.fetch_data(table_name, page, per_page)
    total_original = db.count_rows(table_name)
    
    # Build filters for included data
    filters = {}
    if name_filter:
        filters['name'] = name_filter
    if month_filter:
        filters['birth_month'] = month_filter
    if year_filter:
        filters['birth_year'] = year_filter
    if day_filter:
        filters['birth_day'] = day_filter
    
    # Fetch included data
    included_data = db.fetch_data(included_table, included_page, per_page, filters, sort_by, sort_order)
    total_included = db.count_rows(included_table, filters)
    
    # Fetch excluded data
    excluded_data = db.fetch_data(excluded_table, excluded_page, per_page)
    total_excluded = db.count_rows(excluded_table)
    
    summary_stats = dataset.get('summary_stats')
    
    # Calculate pagination
    total_pages = math.ceil(total_original / per_page) if total_original else 1
    included_total_pages = math.ceil(total_included / per_page) if total_included else 1
    excluded_total_pages = math.ceil(total_excluded / per_page) if total_excluded else 1

    return render_template('index.html',
                           data=csv_data,
                           page=page,
                           total_pages=total_pages,
                           included_data=included_data,
                           excluded_data=excluded_data,
                           included_page=included_page,
                           excluded_page=excluded_page,
                           included_total_pages=included_total_pages,
                           excluded_total_pages=excluded_total_pages,
                           total_included=total_included,
                           total_excluded=total_excluded,
                           summary_stats=summary_stats,
                           name_filter=name_filter,
                           month_filter=month_filter,
                           year_filter=year_filter,
                           day_filter=day_filter,
                           sort_by=sort_by,
                           sort_order=sort_order,
                           datasets=datasets,
                           current_dataset_id=get_current_dataset_id())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        dataset_list = get_dataset_list()
        dataset_id = f"dataset_{len(dataset_list) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filepath = os.path.join(app.config['DATA_FOLDER'], f"{dataset_id}_{filename}")

        file.save(filepath)
        
        # Initialize progress
        set_upload_progress(dataset_id, 0, 100, 'starting')
        
        # Create dataset metadata first
        table_name = db.sanitize_table_name(filename)
        metadata = {
            'table_name': table_name,
            'included_table': f"{table_name}_included",
            'excluded_table': f"{table_name}_excluded",
            'summary_stats': None,
            'filename': filename,
            'filepath': filepath,
            'encoding': None,
            'status': 'uploading'
        }
        
        save_dataset_metadata(dataset_id, metadata)
        dataset_list.append(dataset_id)
        set_dataset_list(dataset_list)
        set_current_dataset_id(dataset_id)
        
        # Start background upload thread
        thread = Thread(target=process_upload_background, args=(dataset_id, filepath, filename))
        thread.daemon = True
        thread.start()
        
        logging.info(f"Started background upload for dataset {dataset_id}: {filename}")
        
        # Redirect immediately to show progress
        return redirect(url_for('index', dataset_id=dataset_id))
    
    return redirect(url_for('index'))

def process_upload_background(dataset_id, filepath, filename):
    """Background thread to process large file uploads"""
    try:
        # Detect encoding
        with open(filepath, 'rb') as f:
            raw_bytes = f.read(4096)
            result = chardet.detect(raw_bytes)
            encoding = result['encoding'] or 'latin-1'
            logging.info(f"Detected encoding: {encoding}")
        
        set_upload_progress(dataset_id, 5, 100, 'scanning')
        
        # Pre-scan for clean/problematic rows
        logging.info("Pre-scanning CSV for column consistency...")
        clean_rows = []
        extra_column_rows = []
        headers = None
        expected_cols = 4
        
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            expected_cols = len(headers)
            
            row_count = 0
            for line_num, row in enumerate(csv_reader, start=2):
                row_count += 1
                
                if len(row) == expected_cols:
                    clean_rows.append(row)
                else:
                    cleaned_row = [cell.replace('\x00', '') if cell else '' for cell in row]
                    
                    first_name = cleaned_row[0] if len(cleaned_row) > 0 else '-'
                    birth_day = cleaned_row[1] if len(cleaned_row) > 1 else ''
                    birth_month = cleaned_row[2] if len(cleaned_row) > 2 else ''
                    birth_year = cleaned_row[3] if len(cleaned_row) > 3 else ''
                    
                    extra_columns = ''
                    if len(cleaned_row) > expected_cols:
                        extra_data = cleaned_row[expected_cols:]
                        extra_columns = json.dumps(extra_data)
                    
                    extra_column_rows.append({
                        'FirstName': first_name,
                        'BirthDay': birth_day,
                        'BirthMonth': birth_month,
                        'BirthYear': birth_year,
                        'extra_columns': extra_columns
                    })
                
                # Update progress every 10000 rows
                if row_count % 10000 == 0:
                    total_so_far = len(clean_rows) + len(extra_column_rows)
                    progress = min(5 + (row_count / max(total_so_far, 1)) * 15, 20)
                    set_upload_progress(dataset_id, int(progress), 100, f'scanning ({row_count:,} rows)')
        
        set_upload_progress(dataset_id, 20, 100, 'processing')
        logging.info(f"Pre-scan complete: {len(clean_rows)} clean rows, {len(extra_column_rows)} rows with extra columns")
        
        # Process with pandas
        csv_data = []
        if len(clean_rows) > 0:
            df = pd.DataFrame(clean_rows, columns=headers)
            
            for col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.replace('\x00', '', regex=False)
            
            col_names = df.columns.tolist()
            first_name_col = col_names[0] if len(col_names) > 0 else None
            birth_day_col = col_names[1] if len(col_names) > 1 else None
            birth_month_col = col_names[2] if len(col_names) > 2 else None
            birth_year_col = col_names[3] if len(col_names) > 3 else None
            
            result_df = pd.DataFrame()
            result_df['FirstName'] = df[first_name_col].fillna('-') if first_name_col else '-'
            result_df['BirthDay'] = df[birth_day_col].fillna('') if birth_day_col else ''
            result_df['BirthMonth'] = df[birth_month_col].fillna('') if birth_month_col else ''
            result_df['BirthYear'] = df[birth_year_col].fillna('') if birth_year_col else ''
            result_df['extra_columns'] = ''
            
            csv_data = result_df.to_dict('records')
        
        set_upload_progress(dataset_id, 30, 100, 'creating table')
        
        # Combine all rows
        all_rows = csv_data + extra_column_rows
        logging.info(f"Total rows to insert: {len(all_rows)}")
        
        # Create table
        table_name = db.sanitize_table_name(filename)
        db.create_original_data_table(table_name)
        
        # Insert with progress callback
        def progress_callback(current, total, stage):
            # Map to 30-100% range
            progress = 30 + int((current / total) * 70)
            set_upload_progress(dataset_id, progress, 100, f'{stage} ({current:,}/{total:,})')
        
        if all_rows:
            db.insert_data(table_name, all_rows, progress_callback=progress_callback)
        
        # Update metadata
        dataset = load_dataset_metadata(dataset_id)
        dataset['encoding'] = encoding
        dataset['status'] = 'ready'
        dataset['pre_excluded_count'] = len(extra_column_rows)
        save_dataset_metadata(dataset_id, dataset)
        
        set_upload_progress(dataset_id, 100, 100, 'complete')
        logging.info(f"Upload complete for dataset {dataset_id}")
        
    except Exception as e:
        logging.error(f"Error in background upload: {e}")
        import traceback
        traceback.print_exc()
        set_upload_progress(dataset_id, 0, 100, f'error: {str(e)}')
        
        # Update metadata
        dataset = load_dataset_metadata(dataset_id)
        if dataset:
            dataset['status'] = 'error'
            save_dataset_metadata(dataset_id, dataset)

@app.route('/api/upload-progress/<dataset_id>')
def get_upload_progress_api(dataset_id):
    """API endpoint to get current upload progress"""
    progress = get_upload_progress(dataset_id)
    if progress:
        return jsonify(progress)
    return jsonify({'current': 0, 'total': 100, 'stage': 'unknown', 'percentage': 0})

@app.route('/clean', methods=['POST'])
def clean_data():
    dataset_id = request.form.get('dataset_id', get_current_dataset_id())
    if not dataset_id:
        logging.error("No dataset ID provided")
        return redirect(url_for('index'))
    
    dataset = load_dataset_metadata(dataset_id)
    if not dataset:
        logging.error("Dataset not found")
        return redirect(url_for('index'))
    
    filepath = dataset['filepath']

    if not os.path.exists(filepath):
        logging.error("File not found for cleaning")
        return redirect(url_for('index'))

    try:
        # Get stored encoding from metadata
        encoding = dataset.get('encoding', 'utf-8')
        logging.info(f"Using encoding for cleaning: {encoding}")
        
        # Use DataCleaner's clean_data_from_file method
        cleaner = DataCleaner()
        included_df, excluded_df = cleaner.clean_data_from_file(filepath, encoding)
        
        # Remove null bytes from included data
        for col in included_df.columns:
            if included_df[col].dtype == 'object':
                included_df[col] = included_df[col].astype(str).str.replace('\x00', '', regex=False)
        
        # Remove null bytes from excluded data
        for col in excluded_df.columns:
            if excluded_df[col].dtype == 'object':
                excluded_df[col] = excluded_df[col].astype(str).str.replace('\x00', '', regex=False)
        
        # Calculate analytics
        analytics = DataAnalytics()
        report_gen = ReportGenerator()
        
        uniqueness = analytics.calculate_uniqueness_metrics(included_df)
        duplicates = analytics.find_duplicate_records(included_df)
        top_80_data = analytics.calculate_top_80_names(included_df)
        
        summary_stats = report_gen.get_summary_stats(
            included_df, excluded_df, cleaner.original_count,
            uniqueness, duplicates, top_80_data
        )
        
        # Create tables and insert data
        included_table = dataset['included_table']
        excluded_table = dataset['excluded_table']
        
        db.create_included_data_table(included_table)
        db.create_excluded_data_table(excluded_table)
        
        if not included_df.empty:
            included_records = included_df.to_dict('records')
            db.insert_data(included_table, included_records)
        
        if not excluded_df.empty:
            excluded_records = excluded_df.to_dict('records')
            db.insert_data(excluded_table, excluded_records)
        
        # Update metadata
        dataset['summary_stats'] = summary_stats
        save_dataset_metadata(dataset_id, dataset)
        
        logging.info(f"Data cleaning completed for {dataset_id}: {len(included_df)} included, {len(excluded_df)} excluded")
        
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        import traceback
        traceback.print_exc()

    return redirect(url_for('index', dataset_id=dataset_id))

@app.route('/clear/<dataset_id>', methods=['POST'])
def clear_dataset(dataset_id):
    dataset_list = get_dataset_list()
    
    if dataset_id in dataset_list:
        dataset = load_dataset_metadata(dataset_id)
        if dataset:
            # Drop Supabase tables
            table_name = dataset.get('table_name')
            if table_name:
                db.drop_table(table_name)
                db.drop_table(f"{table_name}_included")
                db.drop_table(f"{table_name}_excluded")
            
            # Delete file
            filepath = dataset.get('filepath')
            if filepath and os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                except Exception as e:
                    logging.error(f"Error deleting file: {e}")
        
        # Delete metadata
        meta_filepath = os.path.join(CACHE_DIR, f'{dataset_id}_meta.pkl')
        if os.path.exists(meta_filepath):
            try:
                os.unlink(meta_filepath)
            except Exception as e:
                logging.error(f"Error deleting metadata: {e}")
        
        dataset_list.remove(dataset_id)
        set_dataset_list(dataset_list)
        
        current_id = get_current_dataset_id()
        if current_id == dataset_id:
            new_current = dataset_list[0] if dataset_list else None
            set_current_dataset_id(new_current)
    
    return redirect(url_for('index'))

@app.route('/clear', methods=['POST'])
def clear_data():
    dataset_list = get_dataset_list()
    
    for dataset_id in dataset_list:
        dataset = load_dataset_metadata(dataset_id)
        if dataset:
            table_name = dataset.get('table_name')
            if table_name:
                db.drop_table(table_name)
                db.drop_table(f"{table_name}_included")
                db.drop_table(f"{table_name}_excluded")
            
            filepath = dataset.get('filepath')
            if filepath and os.path.exists(filepath):
                try:
                    os.unlink(filepath)
                except Exception as e:
                    logging.error(f"Error deleting file: {e}")
        
        meta_filepath = os.path.join(CACHE_DIR, f'{dataset_id}_meta.pkl')
        if os.path.exists(meta_filepath):
            try:
                os.unlink(meta_filepath)
            except Exception as e:
                logging.error(f"Error deleting metadata: {e}")
    
    set_dataset_list([])
    set_current_dataset_id(None)
    
    return redirect(url_for('index'))

@app.route('/download/included/csv')
def download_included_csv():
    dataset = get_current_dataset()
    if dataset:
        table_name = dataset.get('included_table')
        if table_name:
            data = db.fetch_all_data(table_name)
            included_df = pd.DataFrame(data)
            
            if not included_df.empty:
                output = io.StringIO()
                included_df.to_csv(output, index=False)
                output.seek(0)
                
                response = make_response(output.getvalue())
                filename = dataset.get('filename', 'data').replace('.csv', '')
                response.headers['Content-Disposition'] = f'attachment; filename={filename}_included.csv'
                response.headers['Content-Type'] = 'text/csv'
                return response
    return "No data available", 404

@app.route('/download/included/pdf')
def download_included_pdf():
    dataset = get_current_dataset()
    if not dataset:
        return "No data available", 404
    
    table_name = dataset.get('included_table')
    if not table_name:
        return "No data available", 404
    
    data = db.fetch_all_data(table_name)
    included_df = pd.DataFrame(data)
    
    if included_df.empty:
        return "No data available", 404
    
    summary_stats = dataset.get('summary_stats')
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=1
    )
    elements.append(Paragraph(f"Data Included Report - {dataset.get('filename', 'Unknown')}", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    summary_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
    summary_text += f"Total Records: {len(included_df)}<br/>"
    if summary_stats:
        summary_text += f"Unique Names: {summary_stats['uniqueness']['total_unique_names']}<br/>"
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    data_rows = [['Row ID', 'Name', 'Birth Day', 'Birth Month', 'Birth Year']]
    for idx, row in included_df.iterrows():
        data_rows.append([
            str(row['row_id']),
            str(row['name']),
            str(row['birth_day']),
            str(row['birth_month']),
            str(row['birth_year'])
        ])
    
    table = Table(data_rows, colWidths=[2.5*inch, 2*inch, 1*inch, 1.2*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8)
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    buffer.seek(0)
    filename = dataset.get('filename', 'data').replace('.csv', '')
    return send_file(buffer, as_attachment=True, download_name=f'{filename}_included_report.pdf', mimetype='application/pdf')

@app.route('/download/excluded/csv')
def download_excluded_csv():
    dataset = get_current_dataset()
    if dataset:
        table_name = dataset.get('excluded_table')
        if table_name:
            data = db.fetch_all_data(table_name)
            excluded_df = pd.DataFrame(data)
            
            if not excluded_df.empty:
                output = io.StringIO()
                excluded_df.to_csv(output, index=False)
                output.seek(0)
                
                response = make_response(output.getvalue())
                filename = dataset.get('filename', 'data').replace('.csv', '')
                response.headers['Content-Disposition'] = f'attachment; filename={filename}_excluded.csv'
                response.headers['Content-Type'] = 'text/csv'
                return response
    return "No data available", 404

@app.route('/download/excluded/pdf')
def download_excluded_pdf():
    dataset = get_current_dataset()
    if not dataset:
        return "No data available", 404
    
    table_name = dataset.get('excluded_table')
    if not table_name:
        return "No data available", 404
    
    data = db.fetch_all_data(table_name)
    excluded_df = pd.DataFrame(data)
    
    if excluded_df.empty:
        return "No data available", 404
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#dc3545'),
        spaceAfter=30,
        alignment=1
    )
    elements.append(Paragraph(f"Data Exclusion Report - {dataset.get('filename', 'Unknown')}", title_style))
    elements.append(Spacer(1, 0.3*inch))
    
    summary_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>"
    summary_text += f"Total Excluded Records: {len(excluded_df)}<br/>"
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    data_rows = [['Row ID', 'Name', 'Birth Day', 'Birth Month', 'Birth Year', 'Exclusion Reason']]
    for idx, row in excluded_df.iterrows():
        data_rows.append([
            str(row['row_id']),
            str(row['name']) if row['name'] else '-',
            str(row['birth_day']) if row['birth_day'] else '-',
            str(row['birth_month']) if row['birth_month'] else '-',
            str(row['birth_year']) if row['birth_year'] else '-',
            str(row['exclusion_reason'])
        ])
    
    table = Table(data_rows, colWidths=[2.5*inch, 1.5*inch, 0.8*inch, 1*inch, 0.8*inch, 2.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    elements.append(table)
    doc.build(elements)
    
    buffer.seek(0)
    filename = dataset.get('filename', 'data').replace('.csv', '')
    return send_file(buffer, as_attachment=True, download_name=f'{filename}_excluded_report.pdf', mimetype='application/pdf')

@app.route('/download/top80/csv')
def download_top80_csv():
    dataset = get_current_dataset()
    if not dataset or not dataset.get('summary_stats') or 'top_80_names' not in dataset['summary_stats']:
        return "No data available", 404
    
    top_80_data = dataset['summary_stats']['top_80_names']
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Frequency', 'Percentage'])
    
    for name_info in top_80_data['top_names']:
        writer.writerow([name_info['name'], name_info['frequency'], name_info['percentage']])
    
    output.seek(0)
    response = make_response(output.getvalue())
    filename = dataset.get('filename', 'data').replace('.csv', '')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}_top_80_percent_names.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/download/top80/json')
def download_top80_json():
    dataset = get_current_dataset()
    if not dataset or not dataset.get('summary_stats') or 'top_80_names' not in dataset['summary_stats']:
        return "No data available", 404
    
    top_80_data = dataset['summary_stats']['top_80_names']
    
    output = json.dumps(top_80_data, indent=2)
    response = make_response(output)
    filename = dataset.get('filename', 'data').replace('.csv', '')
    response.headers['Content-Disposition'] = f'attachment; filename={filename}_top_80_percent_names.json'
    response.headers['Content-Type'] = 'application/json'
    return response

@app.route('/api/chart-data')
def get_chart_data():
    dataset = get_current_dataset()
    if dataset:
        table_name = dataset.get('included_table')
        if table_name:
            data = db.fetch_all_data(table_name)
            included_df = pd.DataFrame(data)
            
            if not included_df.empty:
                year_counts = included_df['birth_year'].value_counts().sort_index()
                year_data = {
                    'labels': [str(year) for year in year_counts.index.tolist()],
                    'values': year_counts.values.tolist()
                }
                
                month_counts = included_df['birth_month'].value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                month_data = {
                    'labels': [month_names[int(m)-1] for m in month_counts.index.tolist()],
                    'values': month_counts.values.tolist()
                }
                
                return jsonify({
                    'year_distribution': year_data,
                    'month_distribution': month_data
                })
    
    return jsonify({'error': 'No data available'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)