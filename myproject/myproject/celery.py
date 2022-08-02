from __future__ import absolute_import, unicode_literals
import os

from celery import Celery
from django.conf import settings
from celery.schedules import crontab


os.environ.setdefault('DJANGO_SETTINGS_MODULE', "myproject.settings")

app = Celery("myproject")
app.conf.enable_utc = False

app.conf.update(timezone = "Asia/Kathmandu")

app.config_from_object(settings, namespace="CELERY")

@app.task(bind=True)
def debug_task(self):
    print(f"Request: {self.request!r}")
 

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # Executes every Monday morning at 7:30 a.m.
    sender.add_periodic_task(
        crontab(hour=0, minute=1),
        remove_files.s(),
    )

@app.task
def remove_files():
    files_dir = os.path.join(settings.MEDIA_ROOT, "temp")
    os.system(f"rm -rf {files_dir}")
    print("Delete files")


