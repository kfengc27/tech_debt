from django.urls import path
from . import views
from .views import upload_csv_view, chart_view, compare_view

urlpatterns = [
    path('', views.upload_csv_view, name="upload_csv_view"),
    path("charts/", chart_view, name="chart_view"),
    path("compare/", views.compare_view, name="compare_view"),  # ✅ 新增
    path("projects/", views.projects_view, name="projects_view"),  # 👈 新增
]