NEXUS HARDWARE QUALIFICATION
==============================

1. Open/run NEXUS_Qualification.
   - Windows: double-click NEXUS_Qualification.bat
   - macOS:   double-click NEXUS_Qualification.command
   - Linux:   run ./NEXUS_Qualification.sh
2. First run needs Python 3.10+ and internet ONCE (pinned runtime setup +
   any missing official file). Afterwards the qualification runs fully offline.
3. Connect the laptop charger if the application asks.
4. Do not close the application while the test is running
   (about 10 minutes, plus setup).
5. Wait for the final report (it opens automatically).
6. Send the generated NEXUS_Qualification_....zip file to the buyer.

If the black window closes immediately without testing anything:
open the file nexus_launcher.log (next to the launcher, or on the
Desktop on macOS) — its last lines say exactly what is missing
(Python, internet, disk space, or files not extracted).
Fix that one thing and double-click again. Setup resumes itself.

No files need to be selected.
No benchmark settings need to be changed.
