import streamlit as st
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import tempfile
import os

def create_pdf_report(data, analysis_results, model_results, forecast_results, filename="time_series_analysis.pdf"):
    """
    Create a comprehensive PDF report of the time series analysis
    """
    try:
        # Create a temporary file for the PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf_path = tmp_file.name
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E86C1'),
            alignment=TA_CENTER,
            spaceAfter=30
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1B4F72'),
            spaceAfter=12
        )
        
        normal_style = styles['Normal']
        normal_style.fontSize = 11
        normal_style.spaceAfter = 12
        
        # Build the story (content)
        story = []
        
        # Title page
        story.append(Paragraph("TIME FLOW", title_style))
        story.append(Paragraph("Reporte Completo de Análisis de Series Temporales", styles['Title']))
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        story.append(Paragraph(f"<b>Fecha del Reporte:</b> {report_date}", normal_style))
        story.append(Paragraph(f"<b>Número de Observaciones:</b> {len(data)}", normal_style))
        story.append(Paragraph(f"<b>Período:</b> {data.index.min().strftime('%Y-%m-%d')} a {data.index.max().strftime('%Y-%m-%d')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Resumen Ejecutivo", heading_style))
        story.append(Paragraph(
            "Este reporte presenta un análisis completo de series temporales utilizando la plataforma TIME FLOW. "
            "El análisis incluye preprocesamiento de datos, modelado estadístico y generación de pronósticos.",
            normal_style
        ))
        story.append(PageBreak())
        
        # Data Overview
        story.append(Paragraph("1. Descripción de los Datos", heading_style))
        
        # Data statistics table
        data_stats = pd.DataFrame({
            'Métrica': ['Número de Observaciones', 'Valor Promedio', 'Desviación Estándar', 'Valor Mínimo', 'Valor Máximo'],
            'Valor': [
                len(data),
                f"{data.iloc[:, 0].mean():.2f}",
                f"{data.iloc[:, 0].std():.2f}",
                f"{data.iloc[:, 0].min():.2f}",
                f"{data.iloc[:, 0].max():.2f}"
            ]
        })
        
        # Convert to table
        table_data = [['Métrica', 'Valor']]
        for _, row in data_stats.iterrows():
            table_data.append([row['Métrica'], row['Valor']])
        
        table = Table(table_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF5FB')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 20))
        
        # Data visualization
        story.append(Paragraph("Visualización de la Serie Temporal", heading_style))
        
        # Create time series plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index, data.iloc[:, 0], linewidth=2, color='#2E86C1')
        ax.set_title('Serie Temporal', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Valor')
        ax.grid(True, alpha=0.3)
        
        # Save plot to temporary file
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png', dpi=150, bbox_inches='tight')
        plot_buffer.seek(0)
        
        # Create image
        img = Image(plot_buffer, width=6*inch, height=3.6*inch)
        story.append(img)
        story.append(Spacer(1, 20))
        plt.close()
        
        # Analysis Results
        if analysis_results:
            story.append(PageBreak())
            story.append(Paragraph("2. Resultados del Análisis", heading_style))
            
            # Add analysis interpretation
            if 'stationarity' in analysis_results:
                story.append(Paragraph("Prueba de Estacionariedad", heading_style))
                stationarity = analysis_results['stationarity']
                story.append(Paragraph(f"<b>Estadístico ADF:</b> {stationarity.get('adf_statistic', 'N/A'):.4f}", normal_style))
                story.append(Paragraph(f"<b>Valor p:</b> {stationarity.get('adf_p_value', 'N/A'):.4f}", normal_style))
                
                if stationarity.get('adf_p_value', 1) < 0.05:
                    story.append(Paragraph("La serie es estacionaria (p < 0.05)", normal_style))
                else:
                    story.append(Paragraph("La serie NO es estacionaria (p >= 0.05)", normal_style))
                
                story.append(Spacer(1, 12))
            
            if 'decomposition' in analysis_results:
                story.append(Paragraph("Descomposición de la Serie", heading_style))
                story.append(Paragraph(
                    "La serie temporal ha sido descompuesta en componentes de tendencia, estacionalidad y residuos.",
                    normal_style
                ))
                story.append(Spacer(1, 12))
        
        # Model Results
        if model_results:
            story.append(PageBreak())
            story.append(Paragraph("3. Resultados del Modelado", heading_style))
            
            # Model comparison table
            model_comparison = []
            for model_name, model_result in model_results.items():
                model_comparison.append([
                    model_name,
                    model_result.get('model_type', 'N/A'),
                    f"{model_result.get('aic', 0):.2f}",
                    f"{model_result.get('bic', 0):.2f}"
                ])
            
            if model_comparison:
                table_data = [['Modelo', 'Tipo', 'AIC', 'BIC']]
                table_data.extend(model_comparison)
                
                table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E74C3C')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FADBD8')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 20))
            
            # Best model details
            if model_results:
                best_model_name = min(model_results.keys(), key=lambda x: model_results[x].get('aic', float('inf')))
                best_model = model_results[best_model_name]
                
                story.append(Paragraph(f"Mejor Modelo: {best_model_name}", heading_style))
                story.append(Paragraph(f"<b>Tipo:</b> {best_model.get('model_type', 'N/A')}", normal_style))
                story.append(Paragraph(f"<b>AIC:</b> {best_model.get('aic', 0):.2f}", normal_style))
                story.append(Paragraph(f"<b>BIC:</b> {best_model.get('bic', 0):.2f}", normal_style))
                
                if 'order' in best_model:
                    story.append(Paragraph(f"<b>Orden:</b> {best_model['order']}", normal_style))
                
                story.append(Spacer(1, 12))
        
        # Forecast Results
        if forecast_results:
            story.append(PageBreak())
            story.append(Paragraph("4. Resultados del Pronóstico", heading_style))
            
            # Forecast statistics
            forecasts = forecast_results.get('forecasts', pd.DataFrame())
            
            if not forecasts.empty:
                story.append(Paragraph("Estadísticas del Pronóstico", heading_style))
                
                forecast_stats = pd.DataFrame({
                    'Métrica': ['Pronóstico Promedio', 'Desviación Estándar', 'Valor Mínimo', 'Valor Máximo'],
                    'Valor': [
                        f"{forecasts.iloc[:, 0].mean():.2f}",
                        f"{forecasts.iloc[:, 0].std():.2f}",
                        f"{forecasts.iloc[:, 0].min():.2f}",
                        f"{forecasts.iloc[:, 0].max():.2f}"
                    ]
                })
                
                table_data = [['Métrica', 'Valor']]
                for _, row in forecast_stats.iterrows():
                    table_data.append([row['Métrica'], row['Valor']])
                
                table = Table(table_data, colWidths=[3*inch, 2*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27AE60')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#D5F4E6')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 20))
                
                # Forecast table (first 10 values)
                story.append(Paragraph("Valores Pronosticados", heading_style))
                
                forecast_table_data = [['Período', 'Pronóstico', 'Límite Inferior', 'Límite Superior']]
                for i in range(min(10, len(forecasts))):
                    forecast_table_data.append([
                        str(i+1),
                        f"{forecasts.iloc[i, 0]:.2f}",
                        f"{forecasts.iloc[i, 1]:.2f}" if forecasts.shape[1] > 1 else "N/A",
                        f"{forecasts.iloc[i, 2]:.2f}" if forecasts.shape[1] > 2 else "N/A"
                    ])
                
                table = Table(forecast_table_data, colWidths=[1*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F39C12')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FDF2E9')),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(table)
                story.append(Spacer(1, 20))
        
        # Conclusions
        story.append(PageBreak())
        story.append(Paragraph("5. Conclusiones", heading_style))
        
        conclusions = []
        conclusions.append("• Los datos han sido analizados utilizando técnicas avanzadas de series temporales.")
        
        if model_results:
            best_model_name = min(model_results.keys(), key=lambda x: model_results[x].get('aic', float('inf')))
            conclusions.append(f"• El modelo {best_model_name} mostró el mejor desempeño según el criterio AIC.")
        
        if forecast_results:
            conclusions.append("• Se generaron pronósticos con intervalos de confianza.")
        
        conclusions.append("• Este análisis fue generado automáticamente por TIME FLOW.")
        
        for conclusion in conclusions:
            story.append(Paragraph(conclusion, normal_style))
        
        story.append(Spacer(1, 20))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("---", normal_style))
        story.append(Paragraph("Reporte generado por TIME FLOW - Plataforma de Análisis de Series Temporales", 
                             ParagraphStyle('Footer', parent=normal_style, fontSize=9, textColor=colors.grey, alignment=TA_CENTER)))
        
        # Build PDF
        doc.build(story)
        
        # Read the PDF file
        with open(pdf_path, 'rb') as f:
            pdf_data = f.read()
        
        # Clean up temporary file
        os.unlink(pdf_path)
        
        return pdf_data, None
        
    except Exception as e:
        return None, str(e)

def create_downloadable_pdf(data, analysis_results=None, model_results=None, forecast_results=None):
    """
    Create a downloadable PDF report
    """
    try:
        pdf_data, error = create_pdf_report(data, analysis_results, model_results, forecast_results)
        
        if error:
            return None, error
        
        # Create download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"time_series_analysis_{timestamp}.pdf"
        
        return pdf_data, filename
        
    except Exception as e:
        return None, str(e)