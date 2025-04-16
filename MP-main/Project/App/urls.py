from django.urls import path
from . import views

urlpatterns = [
    # Home page
    path('', views.home, name='home'),

    # Graphical Method URLs
    path('graphical_method/', views.graphical_method, name='graphical_method'),
    path('graphical_method/solve/', views.graphical_solve, name='graphical_solve'),
    path('graphical_method/steps/', views.graphical_steps, name='graphical_steps'),
    path('graphical_method/application/', views.graphical_application, name='graphical_application'),
    # Simplex Method
    path('simplex/', views.simplex_method, name='simplex_method'),
    path('simplex/steps/', views.simplex_steps, name='simplex_steps'),
    path('simplex/solve/', views.simplex_solve, name='simplex_solve'),   
    path('simplex/application/', views.simplex_application, name='simplex_application'),
    path('simplex_solve/', views.simplex_solver, name='simplex_solver'),

    # Transportation Method URLs
    path('transportation_method/', views.transportation_method, name='transportation_method'),
    path('transportation_method/steps/', views.transportation_steps, name='transportation_steps'),
    path('transportation_method/application/', views.transportation_application, name='transportation_application'),
    path('transportation_method/solve/', views.transportation_solve, name='transportation_solve'),

    # Space Mission Planner URLs
    path('terraform/', views.terraform_home, name='terraform_home'),
    path('mission_dashboard/', views.mission_dashboard, name='mission_dashboard'),

]