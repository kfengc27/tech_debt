from django import forms

class UploadCSVForm(forms.Form):
    file = forms.FileField(
        label="Original CSV file",
        widget=forms.ClearableFileInput(attrs={"accept": ".csv"}),
    )

FREQ_CHOICES = [
    ("W", "Weekly"),
    ("M", "Monthly"),
    ("Y", "Yearly"),
]

class ComplexityForm(forms.Form):
    csv_file = forms.FileField(required=True, label="Upload CSV")
    value_col = forms.CharField(required=True, help_text="e.g., complexity_raw")
    date_col = forms.CharField(required=False, help_text="optional: real date column name")
    freq = forms.ChoiceField(choices=FREQ_CHOICES, initial="W", label="Resample by")
    use_fake_dates = forms.BooleanField(required=False, initial=True, label="Use fake dates if no date col?")
    fake_start = forms.DateField(required=False, initial="2021-01-01")
    fake_end = forms.DateField(required=False, initial="2024-12-31")