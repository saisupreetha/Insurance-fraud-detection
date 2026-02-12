from fpdf import FPDF
import pandas as pd
from datetime import datetime
import os

class PDF(FPDF):
    def header(self):
        # Company Letterhead
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 10, 'ACME Insurance Group', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 5, 'Fraud Prevention & Risk Assessment Division', 0, 1, 'C')
        self.cell(0, 5, '123 Insurance Plaza, Claims City, IC 12345 | Phone: (555) 123-4567', 0, 1, 'C')
        self.ln(10)
        
        # Report Title
        self.set_font('Arial', 'B', 14)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, 'Insurance Claim Fraud Risk Assessment Report', 0, 1, 'C')
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, 'This report is confidential and intended for authorized personnel only.', 0, 1, 'C')
        self.cell(0, 5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(claim_data, prediction_prob, risk_level, key_drivers):
    pdf = PDF()
    pdf.add_page()
    
    # Report Metadata
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 8, f"Report ID: FR-{datetime.now().strftime('%Y%m%d%H%M%S')}", ln=True)
    pdf.cell(0, 8, f"Assessment Date: {datetime.now().strftime('%B %d, %Y')}", ln=True)
    pdf.cell(0, 8, f"Analyst: AI Fraud Detection System v2.1", ln=True)
    pdf.ln(10)
    
    # Executive Summary
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "EXECUTIVE SUMMARY", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    
    risk_color = (255, 0, 0) if risk_level == "HIGH RISK" else (255, 165, 0) if risk_level == "MODERATE RISK" else (255, 255, 0) if risk_level == "LOW-MODERATE RISK" else (0, 128, 0)
    risk_description = "High" if risk_level == "HIGH RISK" else "Medium" if risk_level == "MODERATE RISK" else "Low-Medium" if risk_level == "LOW-MODERATE RISK" else "Low"
    
    pdf.set_text_color(*risk_color)
    pdf.cell(0, 8, f"Fraud Risk Level: {risk_description}", ln=True)
    pdf.cell(0, 8, f"Risk Classification: {risk_level}", ln=True)
    pdf.set_text_color(0, 0, 0)
    
    summary_text = f"This automated assessment indicates a {risk_level.lower()} level of fraud suspicion for the submitted claim. "
    if risk_level == "HIGH RISK":
        summary_text += "Immediate investigation is recommended."
    elif risk_level == "MODERATE RISK":
        summary_text += "Enhanced review procedures should be applied."
    else:
        summary_text += "Standard processing may proceed."
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(10)
    
    # Claim Details Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "CLAIM DETAILS", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    
    # Policy Information
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "Policy Information:", ln=True)
    pdf.set_font("Arial", '', 9)
    policy_fields = ['policy_state', 'policy_annual_premium', 'policy_deductable', 'months_as_customer']
    for field in policy_fields:
        if field in claim_data.columns:
            val = claim_data[field].iloc[0]
            label = field.replace('_', ' ').title()
            if 'premium' in field or 'deductable' in field:
                val = f"${val:,.2f}"
            pdf.cell(0, 6, f"  {label}: {val}", ln=True)
    
    pdf.ln(5)
    
    # Incident Details
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "Incident Details:", ln=True)
    pdf.set_font("Arial", '', 9)
    incident_fields = ['incident_type', 'collision_type', 'incident_severity', 'incident_state', 'incident_city', 'authorities_contacted']
    for field in incident_fields:
        if field in claim_data.columns:
            val = claim_data[field].iloc[0]
            label = field.replace('_', ' ').title()
            pdf.cell(0, 6, f"  {label}: {val}", ln=True)
    
    pdf.ln(5)
    
    # Financial Information
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "Financial Information:", ln=True)
    pdf.set_font("Arial", '', 9)
    financial_fields = ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']
    for field in financial_fields:
        if field in claim_data.columns:
            val = claim_data[field].iloc[0]
            label = field.replace('_', ' ').title()
            pdf.cell(0, 6, f"  {label}: ${val:,.2f}", ln=True)
    
    pdf.ln(5)
    
    # Vehicle Information
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "Vehicle Information:", ln=True)
    pdf.set_font("Arial", '', 9)
    vehicle_fields = ['auto_make', 'auto_model', 'auto_year']
    for field in vehicle_fields:
        if field in claim_data.columns:
            val = claim_data[field].iloc[0]
            label = field.replace('_', ' ').title()
            pdf.cell(0, 6, f"  {label}: {val}", ln=True)
    
    pdf.ln(10)
    
    # AI Analysis Section
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "AI RISK ANALYSIS", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    
    pdf.cell(0, 8, f"Assessment Method: AI-Powered Risk Analysis System", ln=True)
    pdf.cell(0, 8, f"Analysis combines machine learning predictions with rule-based heuristics", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 8, "Identified Risk Factors:", ln=True)
    pdf.set_font("Arial", '', 9)
    
    if key_drivers:
        for i, driver in enumerate(key_drivers, 1):
            clean_driver = driver.replace('⚠️', '').strip()
            pdf.cell(0, 6, f"  {i}. {clean_driver}", ln=True)
    else:
        pdf.cell(0, 6, "  No significant heuristic risk factors identified.", ln=True)
    
    pdf.ln(10)
    
    # Investigation Recommendations
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, "INVESTIGATION RECOMMENDATIONS", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 10)
    
    recommendations = []
    if risk_level == "HIGH RISK":
        recommendations = [
            "Immediate escalation to Special Investigations Unit (SIU)",
            "Request independent verification of incident location and timing",
            "Obtain detailed witness statements and contact information",
            "Cross-reference with telematics data if available",
            "Conduct comprehensive background check on claimant"
        ]
    elif risk_level == "MODERATE RISK":
        recommendations = [
            "Perform enhanced desk review with additional documentation requests",
            "Verify police report details directly with law enforcement",
            "Request medical records to corroborate injury claims",
            "Check for consistency in claim narrative across all statements",
            "Consider surveillance if claim value justifies cost"
        ]
    else:
        recommendations = [
            "Proceed with standard claims processing procedures",
            "Monitor claim for any unusual payment patterns",
            "File report for future reference in fraud pattern analysis"
        ]
    
    for rec in recommendations:
        pdf.multi_cell(0, 6, f"- {rec}")
    
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 5, "DISCLAIMER: This assessment is generated by automated systems and should be used as a guide for human review. Final decisions regarding claim validity remain the responsibility of qualified claims adjusters and management. This report does not constitute legal advice.")
    
    # Save
    if not os.path.exists("reports"):
        os.makedirs("reports")
        
    filename = f"reports/fraud_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename
