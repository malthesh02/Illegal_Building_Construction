# report_generator.py (Modified Version)
import os
import json
import logging
from datetime import datetime
from fpdf import FPDF
from typing import Dict, List, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """
    PDF Report Generator for Building Detection Results
    Generates comprehensive reports with images, statistics, and building details
    """
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        # Crucial check: Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_report(self, session_id: str, results: Dict[str, Any], images_dict: Dict[str, str]) -> str:
        """
        Generate a comprehensive PDF report for building detection results
        
        Args:
            session_id: Unique session identifier
            results: Detection results dictionary
            images_dict: Dictionary of image descriptions and file paths
            
        Returns:
            Path to the generated PDF file
        """
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Add cover page
            self._add_cover_page(pdf, session_id, results)
            
            # Add summary section
            self._add_summary_section(pdf, results)
            
            # Add statistics section
            self._add_statistics_section(pdf, results)
            
            # Add images section
            self._add_images_section(pdf, images_dict) # <-- Robust error handling is here
            
            # Add buildings details section
            self._add_buildings_section(pdf, results)
            
            # Add methodology section
            self._add_methodology_section(pdf, results)
            
            # Save PDF
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"BuildingDetectionReport_{session_id}_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # This is line 60, where the error originated.
            pdf.output(filepath)
            logger.info(f"✅ PDF report generated: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error generating PDF report: {e}", exc_info=True)
            # Re-raise to ensure the Flask app catches it and returns the 500
            raise
    
    def _add_cover_page(self, pdf: FPDF, session_id: str, results: Dict[str, Any]):
        """Add cover page to the PDF"""
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 40, 'BUILDING DETECTION REPORT', 0, 1, 'C')
        pdf.ln(20)
        
        # Session info
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 15, 'Detection Analysis Summary', 0, 1, 'C')
        pdf.ln(15)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, f'Session ID: {session_id}', 0, 1, 'C')
        pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        
        mode = results.get('mode', 'single-image')
        mode_display = 'Single Image Detection' if mode == 'single-image' else 'Two-Temporal Change Detection'
        pdf.cell(0, 8, f'Detection Mode: {mode_display}', 0, 1, 'C')
        
        # GPS status
        gps_enabled = results.get('gps_enabled', False)
        gps_status = '🌍 GPS-Enabled' if gps_enabled else '📍 Pixel-Based'
        pdf.cell(0, 8, f'Coordinate System: {gps_status}', 0, 1, 'C')
        
        pdf.ln(20)
        
        # Summary box
        stats = results.get('statistics', {})
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(20, pdf.get_y(), 170, 40, 'F')
        
        pdf.set_xy(25, pdf.get_y() + 5)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 8, 'QUICK STATS', 0, 1)
        
        pdf.set_xy(25, pdf.get_y() + 2)
        pdf.set_font('Arial', '', 10)
        pdf.cell(45, 6, f"Total Buildings: {stats.get('num_buildings_detected', 0)}", 0, 0)
        pdf.cell(45, 6, f"Legal: {stats.get('num_legal_buildings', 0)}", 0, 0)
        pdf.cell(45, 6, f"Illegal: {stats.get('num_illegal_buildings', 0)}", 0, 1)
        
        pdf.set_xy(25, pdf.get_y() + 2)
        pdf.cell(45, 6, f"Compliance: {stats.get('compliance_rate', 0):.1f}%", 0, 0)
        pdf.cell(45, 6, f"Total Area: {stats.get('total_building_area_m2', 0):,.0f} m²", 0, 0)
    
    def _add_summary_section(self, pdf: FPDF, results: Dict[str, Any]):
        """Add executive summary section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '1. Executive Summary', 0, 1)
        pdf.ln(5)
        
        stats = results.get('statistics', {})
        mode = results.get('mode', 'single-image')
        
        pdf.set_font('Arial', '', 12)
        
        # Mode-specific summary
        if mode == 'two-temporal-change-detection':
            pdf.multi_cell(0, 6, 
                f"This report presents the results of a two-temporal building change detection analysis. "
                f"The analysis compared satellite/aerial imagery from two different time periods to identify "
                f"new construction, demolished buildings, and modifications to existing structures."
            )
            pdf.ln(5)
            
            # Change statistics
            changed = stats.get('num_changed_buildings', 0)
            unchanged = stats.get('num_unchanged_buildings', 0)
            change_percent = stats.get('change_percentage', 0)
            
            pdf.multi_cell(0, 6,
                f"Key findings include the detection of {changed} changed buildings ({change_percent:.1f}% change rate) "
                f"and {unchanged} unchanged buildings. The analysis employed K-means clustering and computer vision "
                f"techniques to accurately identify and classify building changes."
            )
        else:
            pdf.multi_cell(0, 6,
                f"This report presents the results of a comprehensive building detection analysis conducted "
                f"on a single satellite/aerial image. The analysis identified {stats.get('num_buildings_detected', 0)} "
                f"buildings and assessed their legal compliance status based on cadastral permit data."
            )
            pdf.ln(5)
            
            pdf.multi_cell(0, 6,
                f"The system detected {stats.get('num_illegal_buildings', 0)} potentially illegal constructions "
                f"representing {stats.get('illegal_building_area_m2', 0):,.0f} m² of unauthorized building area. "
                f"Overall compliance rate for the analyzed area is {stats.get('compliance_rate', 0):.1f}%."
            )
        
        pdf.ln(10)
        
        # Compliance status
        legal_count = stats.get('num_legal_buildings', 0)
        illegal_count = stats.get('num_illegal_buildings', 0)
        uncertain_count = stats.get('num_uncertain_buildings', 0)
        total = legal_count + illegal_count + uncertain_count
        
        if total > 0:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'Compliance Status:', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            pdf.cell(60, 6, f"✅ Legal Buildings: {legal_count} ({legal_count/total*100:.1f}%)", 0, 1)
            pdf.cell(60, 6, f"❌ Illegal Buildings: {illegal_count} ({illegal_count/total*100:.1f}%)", 0, 1)
            pdf.cell(60, 6, f"⚠️  Uncertain Status: {uncertain_count} ({uncertain_count/total*100:.1f}%)", 0, 1)
    
    def _add_statistics_section(self, pdf: FPDF, results: Dict[str, Any]):
        """Add detailed statistics section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '2. Detailed Statistics', 0, 1)
        pdf.ln(5)
        
        stats = results.get('statistics', {})
        mode = results.get('mode', 'single-image')
        
        # Detection Statistics
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Detection Statistics', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        statistics_data = [
            ('Total Buildings Detected', f"{stats.get('num_buildings_detected', 0):,}"),
            ('Legal Buildings', f"{stats.get('num_legal_buildings', 0):,}"),
            ('Illegal Buildings', f"{stats.get('num_illegal_buildings', 0):,}"),
            ('Uncertain Buildings', f"{stats.get('num_uncertain_buildings', 0):,}"),
            ('Compliance Rate', f"{stats.get('compliance_rate', 0):.1f}%"),
            ('Total Building Area', f"{stats.get('total_building_area_m2', 0):,.0f} m²"),
            ('Illegal Building Area', f"{stats.get('illegal_building_area_m2', 0):,.0f} m²")
        ]
        
        if mode == 'two-temporal-change-detection':
            statistics_data.extend([
                ('Changed Buildings', f"{stats.get('num_changed_buildings', 0):,}"),
                ('Unchanged Buildings', f"{stats.get('num_unchanged_buildings', 0):,}"),
                ('Overall Change %', f"{stats.get('change_percentage', 0):.1f}%")
            ])
        
        for label, value in statistics_data:
            pdf.cell(90, 6, f"{label}:", 0, 0)
            pdf.cell(0, 6, value, 0, 1)
        
        pdf.ln(10)
        
        # Area Analysis
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Area Analysis', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        total_area = stats.get('total_building_area_m2', 0)
        illegal_area = stats.get('illegal_building_area_m2', 0)
        legal_area = total_area - illegal_area
        
        if total_area > 0:
            pdf.cell(0, 6, f"Total Built Area: {total_area:,.0f} m²", 0, 1)
            pdf.cell(0, 6, f"Authorized Area: {legal_area:,.0f} m² ({legal_area/total_area*100:.1f}%)", 0, 1)
            pdf.cell(0, 6, f"Unauthorized Area: {illegal_area:,.0f} m² ({illegal_area/total_area*100:.1f}%)", 0, 1)
        
        # GPS Information
        if results.get('gps_enabled') and results.get('image_bounds'):
            pdf.ln(10)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, 'GPS Coverage Area', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            bounds = results['image_bounds']
            pdf.cell(0, 6, f"North: {bounds.get('north', 'N/A')}°", 0, 1)
            pdf.cell(0, 6, f"South: {bounds.get('south', 'N/A')}°", 0, 1)
            pdf.cell(0, 6, f"East: {bounds.get('east', 'N/A')}°", 0, 1)
            pdf.cell(0, 6, f"West: {bounds.get('west', 'N/A')}°", 0, 1)
    
    def _add_images_section(self, pdf: FPDF, images_dict: Dict[str, str]):
        """Add images section to the PDF"""
        if not images_dict:
            return
            
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '3. Visual Results', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        
        for description, image_path in images_dict.items():
            if not image_path:
                logger.warning(f"⚠️ Image path for '{description}' is empty.")
                continue
                
            # Check existence again (robustness)
            if not os.path.exists(image_path):
                logger.error(f"❌ Image file NOT FOUND: {image_path}")
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, f"{description} (Error: File Not Found)", 0, 1)
                pdf.set_font('Arial', 'I', 8)
                pdf.multi_cell(0, 4, f"Path: {image_path}. Check file existence and path resolution in app.py.", 0)
                pdf.ln(10)
                continue
                
            try:
                # Add image description
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 8, description, 0, 1)
                pdf.set_font('Arial', 'I', 8)
                pdf.cell(0, 4, f"File: {os.path.basename(image_path)}", 0, 1)
                pdf.ln(2)
                
                # --- CRITICAL FIX: Robustly handle image embedding failure ---
                try:
                    # Add image (resize to fit page)
                    pdf.image(image_path, x=10, y=pdf.get_y(), w=190)
                except Exception as img_error:
                    # Log the critical failure, but continue
                    logger.critical(f"❌ FPDF FAILED TO EMBED IMAGE '{description}' at path: {image_path}. Error: {img_error}", exc_info=True)
                    
                    # Add a placeholder box and text in the PDF
                    pdf.set_font('Arial', 'B', 10)
                    pdf.set_fill_color(255, 200, 200) # Light red background
                    pdf.rect(10, pdf.get_y(), 190, 80, 'F')
                    pdf.set_xy(15, pdf.get_y() + 5)
                    pdf.cell(0, 10, f"IMAGE LOADING FAILED: {description}", 0, 1)
                    pdf.set_font('Arial', '', 8)
                    pdf.set_xy(15, pdf.get_y())
                    pdf.multi_cell(0, 4, f"File: {image_path}\nError: {str(img_error)}", 0)
                    
                pdf.ln(85)  # Space after image (regardless of if image was loaded or placeholder used)
                
                # Add page break if needed
                if pdf.get_y() > 250:
                    pdf.add_page()
                    
            except Exception as e:
                # Catch general errors during the loop iteration
                logger.error(f"❌ General error processing image {image_path}: {e}")
                pdf.cell(0, 6, f"Error processing image section for {description}: {str(e)}", 0, 1)
                pdf.ln(5)
    
    def _add_buildings_section(self, pdf: FPDF, results: Dict[str, Any]):
        """Add building details section"""
        buildings = results.get('buildings', [])
        if not buildings:
            return
            
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '4. Building Details', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 9)
        
        # Table header
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(15, 8, 'ID', 1, 0, 'C', 1)
        pdf.cell(25, 8, 'Status', 1, 0, 'C', 1)
        pdf.cell(25, 8, 'Area (m²)', 1, 0, 'C', 1)
        pdf.cell(25, 8, 'Confidence', 1, 0, 'C', 1)
        pdf.cell(50, 8, 'GPS Coordinates', 1, 0, 'C', 1)
        pdf.cell(50, 8, 'Details', 1, 1, 'C', 1)
        
        # Building rows
        for i, building in enumerate(buildings[:50]):  # Limit to first 50 buildings
            if pdf.get_y() > 270:  # Add new page if near bottom
                pdf.add_page()
                # Add header again
                pdf.set_fill_color(200, 200, 200)
                pdf.cell(15, 8, 'ID', 1, 0, 'C', 1)
                pdf.cell(25, 8, 'Status', 1, 0, 'C', 1)
                pdf.cell(25, 8, 'Area (m²)', 1, 0, 'C', 1)
                pdf.cell(25, 8, 'Confidence', 1, 0, 'C', 1)
                pdf.cell(50, 8, 'GPS Coordinates', 1, 0, 'C', 1)
                pdf.cell(50, 8, 'Details', 1, 1, 'C', 1)
            
            # Building data
            building_id = building.get('id', f'B{i+1}')
            
            # Status with color
            status = building.get('legal_status', 'UNKNOWN')
            if status == 'LEGAL':
                status_display = '✅ LEGAL'
            elif status == 'ILLEGAL':
                status_display = '❌ ILLEGAL'
            else:
                status_display = '⚠️ UNCERTAIN'
            
            # Area
            area_m2 = building.get('area_m2') or building.get('area_pixels', 0) * 0.25
            area_display = f"{area_m2:.0f}" if area_m2 else "N/A"
            
            # Confidence
            confidence = building.get('confidence', 0)
            confidence_display = f"{confidence:.0f}%" if confidence else "N/A"
            
            # GPS coordinates
            lat = building.get('gps_latitude')
            lon = building.get('gps_longitude')
            if lat is not None and lon is not None:
                gps_display = f"{lat:.4f}, {lon:.4f}"
            else:
                gps_display = "No GPS"
            
            # Details (truncated if too long)
            details = building.get('legal_reason', '') or building.get('change_status', '')
            if len(details) > 30:
                details = details[:27] + "..."
            
            # Add row
            pdf.cell(15, 8, str(building_id), 1, 0, 'C')
            pdf.cell(25, 8, status_display, 1, 0, 'C')
            pdf.cell(25, 8, area_display, 1, 0, 'C')
            pdf.cell(25, 8, confidence_display, 1, 0, 'C')
            pdf.cell(50, 8, gps_display, 1, 0, 'C')
            pdf.cell(50, 8, details, 1, 1, 'C')
        
        if len(buildings) > 50:
            pdf.ln(5)
            pdf.cell(0, 8, f"... and {len(buildings) - 50} more buildings", 0, 1)
    
    def _add_methodology_section(self, pdf: FPDF, results: Dict[str, Any]):
        """Add methodology and technical details section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '5. Methodology & Technical Details', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Detection Methodology', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        mode = results.get('mode', 'single-image')
        
        if mode == 'two-temporal-change-detection':
            pdf.multi_cell(0, 6,
                "The two-temporal change detection employs a multi-stage approach:\n\n"
                "1. **Image Preprocessing**: Both images are normalized, resized, and aligned for comparison\n"
                "2. **K-means Clustering**: Unsupervised clustering identifies changed regions\n"
                "3. **Building Segmentation**: Deep learning models detect building footprints\n"
                "4. **Change Analysis**: Building-level changes are quantified and classified\n"
                "5. **Legal Compliance Check**: Detected buildings are cross-referenced with cadastral permits"
            )
        else:
            pdf.multi_cell(0, 6,
                "The single-image building detection employs advanced computer vision techniques:\n\n"
                "1. **Image Preprocessing**: Input image is normalized and enhanced\n"
                "2. **Building Segmentation**: Deep neural networks identify building footprints\n"
                "3. **Feature Extraction**: Building properties (area, shape, location) are calculated\n"
                "4. **GPS Georeferencing**: Pixel coordinates are converted to geographic coordinates\n"
                "5. **Legal Compliance Analysis**: Buildings are checked against permit database"
            )
        
        pdf.ln(10)
        
        # Technical specifications
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Technical Specifications', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        tech_specs = [
            ('Processing Engine', 'Python + OpenCV + Deep Learning'),
            ('GPS Reference', 'WGS84 Coordinate System'),
            ('Area Calculation', 'Pixel-to-meter conversion based on image resolution'),
            ('Confidence Threshold', 'Minimum 70% detection confidence'),
            ('Change Detection', 'K-means clustering with morphological operations'),
            ('Report Generation', 'Automated PDF with embedded analytics')
        ]
        
        for spec, value in tech_specs:
            pdf.cell(60, 6, f"{spec}:", 0, 0)
            pdf.cell(0, 6, value, 0, 1)
        
        pdf.ln(10)
        
        # Disclaimer
        pdf.set_font('Arial', 'I', 8)
        pdf.multi_cell(0, 4,
            "Disclaimer: This report is generated automatically by the Building Detection System. "
            "The results should be verified by qualified professionals before taking any legal actions. "
            "GPS coordinates and area calculations are estimates based on image analysis."
        )


# Utility function for report generation
def generate_detection_report(session_data: Dict[str, Any], output_path: str) -> str:
    """
    Utility function to generate a quick detection report
    
    Args:
        session_data: Detection results data
        output_path: Where to save the report
        
    Returns:
        Path to generated report
    """
    generator = PDFReportGenerator(os.path.dirname(output_path))
    
    # Extract images from session data
    images_dict = {}
    if 'original_image' in session_data:
        images_dict['Original Image'] = session_data['original_image']
    if 'segmentation_result' in session_data:
        images_dict['Building Segmentation'] = session_data['segmentation_result']
    if 'detection_overlay' in session_data:
        images_dict['Detection Results'] = session_data['detection_overlay']
    
    session_id = session_data.get('session_id', 'unknown_session')
    
    return generator.generate_report(session_id, session_data, images_dict)