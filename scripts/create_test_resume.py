from fpdf import FPDF

def create_marketing_resume():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Jane Marketing", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "jane.marketing@email.com | +1 234 567 890 | New York, NY", ln=True, align="C")
    
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Experience", ln=True)
    pdf.set_font("Arial", "", 10)
    
    roles = [
        ("Senior Marketing Manager, TechCorp", "2022 - Present", "Led digital strategy for B2B SaaS, increasing leads by 40%. Managed a budget of $500k and a team of 5."),
        ("Marketing Specialist, Growthly", "2020 - 2022", "Managed social media campaigns and SEO, growing organic traffic by 2x. Improved conversion rate by 15%."),
        ("Marketing Coordinator, StartUp Inc", "2018 - 2020", "Coordinated events and email marketing for early stage growth. Increased open rates by 20%.")
    ]
    
    for role, date, desc in roles:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 5, f"{role} ({date})", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 5, desc)
        pdf.ln(2)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Education", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "BS in Marketing, NYU", ln=True)

    pdf.output("data/marketing_test.pdf")

if __name__ == "__main__":
    create_marketing_resume()
