# Generated by Django 4.1.5 on 2023-04-03 00:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_carro'),
    ]

    operations = [
        migrations.CreateModel(
            name='Car_User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_2', models.CharField(max_length=50)),
                ('car_model_2', models.CharField(max_length=100)),
                ('loan_amount', models.FloatField()),
                ('start_date_2', models.DateTimeField()),
                ('end_date_2', models.DateField()),
                ('interest_rate_2', models.FloatField()),
            ],
        ),
    ]