from django.core.management.base import BaseCommand

from dashboard.services.updater import update_all_repositories


class Command(BaseCommand):
    help = "Check and update all tracked repositories"

    def handle(self, *args, **options):
        result = update_all_repositories()

        self.stdout.write(
            self.style.SUCCESS(
                f"Completed: "
                f"{result['total']} checked, "
                f"{result['updated']} updated, "
                f"{result['no_change']} unchanged, "
                f"{result['failed']} failed."
            )
        )