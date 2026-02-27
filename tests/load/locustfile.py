"""Locust load testing configuration."""
from locust import HttpUser, task

class CropFreshUser(HttpUser):
    @task
    def health_check(self):
        self.client.get("/api/health")
