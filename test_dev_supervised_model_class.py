#!/usr/bin/env python3
import unittest

import pandas as pd
import numpy as np

from healthcareai import DevelopSupervisedModel
from healthcareai.tests.helpers import fixture


class TestErrorHandlingOnIncorrectColumnsForModels(unittest.TestCase):
    # Test the choice of model and the chosen column's data type
    def test_raise_error_on_regression_with_binary_integer_column(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        # build a synthetic column with binary numbers
        predictedcol = 'binary_integer_column'
        df[predictedcol] = np.random.choice([1,2], df.shape[0])

        np.random.seed(42)
        try:
            o = DevelopSupervisedModel(modeltype='regression',
                                       df=df,
                                       predictedcol=predictedcol,
                                       impute=True)
        except RuntimeError as e:
            correct_error = 'Regression requires a numeric column with continuous numeric data. The predicted column %s is a binary column.' % predictedcol
            self.assertEqual(correct_error, e.args[0])
        else:
            self.fail('No error raised.')

    def test_raise_error_on_regression_with_non_numeric_column(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        df['NonNumericColumn'] = 'some text here'

        np.random.seed(42)
        predictedcol = 'NonNumericColumn'
        try:
            o = DevelopSupervisedModel(modeltype='regression',
                                       df=df,
                                       predictedcol=predictedcol,
                                       impute=True)
        except RuntimeError as e:
            correct_error = 'Regression requires a numeric column with continuous numeric data. The predicted column %s is not a numeric column.' % predictedcol
            self.assertEqual(correct_error, e.args[0])
        else:
            self.fail('No error raised.')

    def test_raise_error_on_classification_with_numeric_column(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        predictedcol = 'A1CNBR'
        np.random.seed(42)
        try:
            o = DevelopSupervisedModel(modeltype='classification',
                                        df=df,
                                        predictedcol=predictedcol,
                                        impute=True)
        except RuntimeError as e:
            correct_error = ('Classification requires a binary column with "Y" or "N" values. The predicted column %s is not a binary column.' % predictedcol)
            self.assertEqual(correct_error, e.args[0])
        else:
            self.fail('No error raised.')


class TestDataframeAndColumnValidation(unittest.TestCase):
    # Test the validation function alone
    df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])

    def test_raise_error_on_null_dataframe(self):
        self.assertEqual(
            'There may be a problem. Your dataframe is null.',
            DevelopSupervisedModel.validate_dataframe_and_column(dataframe=None, column='GenderFLG'))

    def test_raise_error_on_null_column(self):
        self.assertEqual(
            'There may be a problem. Your column is null.',
            DevelopSupervisedModel.validate_dataframe_and_column(dataframe=self.df, column=None))

    def test_raise_error_on_string_type(self):
        self.assertEqual(
            'There may be a problem. You did not pass in a dataframe.',
            DevelopSupervisedModel.validate_dataframe_and_column(dataframe='garbage', column='GenderFLG'))

    def test_raise_error_on_non_existent_column(self):
        non_existant_column = 'fake_column'
        self.assertEqual(
            'There may be a problem. The column %s does not exist.' % non_existant_column,
            DevelopSupervisedModel.validate_dataframe_and_column(dataframe=self.df, column=non_existant_column))

    def test_successful_validation(self):
        self.assertTrue(DevelopSupervisedModel.validate_dataframe_and_column(dataframe=self.df, column='GenderFLG'))


class TestDataframeAndColumnValidationInContextOfModels(unittest.TestCase):
    # Test the validation function in context of developing a model
    def test_raise_error_on_null_dataframe_on_classification_and_regression(self):
        for model in ['regression', 'classification']:
            try:
                o = DevelopSupervisedModel(modeltype=model,
                                           df=None,
                                           predictedcol=None,
                                           impute=True)
            except RuntimeError as e:
                correct_error = 'There may be a problem. Your dataframe is null.'
                self.assertEqual(correct_error, e.args[0])
            else:
                self.fail('No error raised.')

    def test_raise_error_on_non_dataframe_on_classification_and_regression(self):
        not_a_dataframe = [1,2,3,4,5]
        for model in ['regression', 'classification']:
            try:
                o = DevelopSupervisedModel(modeltype=model,
                                           df=not_a_dataframe,
                                           predictedcol='Not a real column',
                                           impute=True)
            except RuntimeError as e:
                correct_error = 'There may be a problem. You did not pass in a dataframe.'
                self.assertEqual(correct_error, e.args[0])
            else:
                self.fail('No error raised.')

    def test_raise_error_on_null_column_on_classification_and_regression(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        for model in ['regression', 'classification']:
            try:
                o = DevelopSupervisedModel(modeltype=model,
                                           df=df,
                                           predictedcol=None,
                                           impute=True)
            except RuntimeError as e:
                correct_error = 'There may be a problem. Your column is null.'
                self.assertEqual(correct_error, e.args[0])
            else:
                self.fail('No error raised.')

    def test_raise_error_on_non_existent_column_on_classification_and_regression(self):
        df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
        fake_column_name = 'NotARealColumn'
        for model in ['regression', 'classification']:
            try:
                o = DevelopSupervisedModel(modeltype=model,
                                           df=df,
                                           predictedcol=fake_column_name,
                                           impute=True)
            except RuntimeError as e:
                correct_error = ('There may be a problem. The column %s does not exist.' % fake_column_name)
                self.assertEqual(correct_error, e.args[0])
            else:
                self.fail('No error raised.')


class TestBinaryColumnChecking(unittest.TestCase):
    """
    Note that this method does not check for the validation errors as those are fully tested above
    """
    df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
    # generate some test columns
    df['binary_float_column'] = np.random.choice([1.11111,2.22222], df.shape[0])
    df['binary_integer_column'] = np.random.choice([1,2], df.shape[0])
    df['integer_column'] = np.random.choice([1,2,3], df.shape[0])

    def test_gender_column_is_binary(self):
        self.assertTrue(DevelopSupervisedModel.is_column_binary(dataframe=self.df, column='GenderFLG'))

    def test_binary_float_column_is_binary(self):
        self.assertTrue(DevelopSupervisedModel.is_column_binary(dataframe=self.df, column='binary_float_column'))

    def test_binary_integer_column_is_binary(self):
        self.assertTrue(DevelopSupervisedModel.is_column_binary(dataframe=self.df, column='binary_integer_column'))

    def test_integer_column_is_not_binary(self):
        self.assertFalse(DevelopSupervisedModel.is_column_binary(dataframe=self.df, column='integer_column'))

    def test_patient_id_column_is_not_binary(self):
        self.assertFalse(DevelopSupervisedModel.is_column_binary(dataframe=self.df, column='PatientID'))


class TestNumericColumnChecking(unittest.TestCase):
    """
    Note that this method does not check for the validation errors as those are fully tested above
    """
    df = pd.read_csv(fixture('HCPyDiabetesClinical.csv'),
                         na_values=['None'])
    # generate some test columns
    df['integer_column'] = np.random.choice([1,2], df.shape[0])
    df['float_column'] = np.random.choice([1.11111,2.22222], df.shape[0])
    df['text_column'] = np.random.choice(['text', 'some more text', 'more text', 'again, more'], df.shape[0])

    def test_systolic_column_is_numeric(self):
        self.assertTrue(DevelopSupervisedModel.is_column_numeric(dataframe=self.df, column='SystolicBPNBR'))

    def test_patient_gender_column_non_numeric(self):
        self.assertFalse(DevelopSupervisedModel.is_column_numeric(dataframe=self.df, column='GenderFLG'))

    def test_integer_is_numeric(self):
        self.assertTrue(DevelopSupervisedModel.is_column_numeric(dataframe=self.df, column='integer_column'))

    def test_float_is_numeric(self):
        self.assertTrue(DevelopSupervisedModel.is_column_numeric(dataframe=self.df, column='float_column'))

    def test_text_is_not_numeric(self):
        self.assertFalse(DevelopSupervisedModel.is_column_numeric(dataframe=self.df, column='text_column'))


if __name__ == '__main__':
    unittest.main()
