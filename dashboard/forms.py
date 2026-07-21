from django import forms

from .models import Repository


class RepositoryForm(forms.ModelForm):
    class Meta:
        model = Repository

        fields = [
            "github_url",
        ]

        widgets = {
            "github_url": forms.URLInput(
                attrs={
                    "class": "form-control",
                    "placeholder": (
                        "https://github.com/owner/repository"
                    ),
                    "autocomplete": "off",
                }
            )
        }

    def clean_github_url(self):
        github_url = self.cleaned_data["github_url"]

        normalised_url = Repository.normalise_github_url(
            github_url
        )

        duplicate_query = Repository.objects.filter(
            github_url=normalised_url
        )

        if self.instance.pk:
            duplicate_query = duplicate_query.exclude(
                pk=self.instance.pk
            )

        if duplicate_query.exists():
            raise forms.ValidationError(
                "This GitHub repository is already in the project list."
            )

        return normalised_url