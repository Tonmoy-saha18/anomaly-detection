import pandas as pd
from pandas import DataFrame
import numpy as np
import random
from datetime import datetime, timedelta


def generate_date_time(start) -> datetime:
    time_stamp = start + timedelta(seconds=random.randint(0, 120))
    if (time_stamp.time().hour == 22 and time_stamp.time().minute > 00) or time_stamp.time().hour < 6:
        if time_stamp.time().hour < 6:
            return time_stamp + timedelta(hours=abs(time_stamp.time().hour - 8), seconds=random.randint(0, 120))
        else:
            return time_stamp + timedelta(hours=(23 - time_stamp.time().hour) + 8, seconds=random.randint(0, 120))
    return time_stamp


def generate_anomaly_date_time(start) -> datetime:
    start_second = (23 - start.time().hour)*3600
    end_second = (23 - start.time().hour)*3600 + 7*3600
    return start + timedelta(seconds=random.randint(start_second, end_second))


def is_generate_anomaly() -> bool:
    return np.random.uniform(0, 1) > 0.9


def type_of_anomaly() -> int:
    return random.randint(0, 1)


def get_customer_amount_range_bin():
    num = np.random.uniform(0, 1)
    if num >= 0.95:
        return 4
    elif num >= 0.8:
        return 3
    elif num >= 0.6:
        return 2
    elif num > 0.25:
        return 1
    return 0


def get_agent_amount_range_bin():
    num = np.random.uniform(0, 1)
    if num >= 0.9:
        return 4
    elif num >= 0.75:
        return 3
    elif num >= 0.55:
        return 2
    elif num > 0.25:
        return 1
    return 0


def get_merchant_amount_range_bin():
    num = np.random.uniform(0, 1)
    if num >= 0.95:
        return 5
    elif num >= 0.85:
        return 4
    elif num >= 0.65:
        return 3
    elif num >= 0.45:
        return 2
    elif num > 0.2:
        return 1
    return 0


def getAllTypeOfAccountList(df) -> tuple:
    print("Getting all account list....")
    customer_info = df[df['transaction_code'] == "CW2CW"]
    customer_info1 = df[df['transaction_code'] == "CW2AW"]
    customer_info2 = df[df['transaction_code'] == "CW2MW"]
    customer_info3 = df[df['transaction_code'] == "CW2CB"]
    customer_info4 = df[df['transaction_code'] == "CB2CW"]
    customer_info5 = df[df['transaction_code'] == "CC2CW"]
    customer_accounts = pd.concat(
        [customer_info['from_internal_account'], customer_info['to_internal_account'],
         customer_info1['from_internal_account'],
         customer_info2['from_internal_account'], customer_info3['from_internal_account'],
         customer_info4['to_internal_account'], customer_info5['to_internal_account']]).unique().tolist()

    agent_info1 = df[df['transaction_code'] == 'CW2AW']
    agent_info2 = df[df['transaction_code'] == 'AW2CW']
    agent_accounts = pd.concat(
        [agent_info1['to_internal_account'], agent_info2['from_internal_account']]).unique().tolist()

    merchant_accounts = df[df['transaction_code'] == 'CW2MW']['to_account'].unique().tolist()

    bank_info1 = df[df['transaction_code'] == 'CW2CB']
    bank_info2 = df[df['transaction_code'] == 'CB2CW']
    bank_accounts = pd.concat(
        [bank_info1['to_internal_account'], bank_info2['from_internal_account']]).unique().tolist()

    card_numbers = df[df['transaction_code'] == 'CC2CW']['from_account'].unique().tolist()

    return customer_accounts, agent_accounts, merchant_accounts, bank_accounts, card_numbers


def initializeAccountBalance(accounts, initiation_range, dic) -> dict:
    print("Initializing the account balance....")
    for account in accounts:
        amount = random.uniform(*initiation_range)
        dic[account] = amount
    return dic


def generateCustomerToCustomerTransaction(customer_account_list, account_to_balance_map) -> list[list[str]]:
    global count
    global start_time
    print("Generating CW2CW transaction.....")
    transaction_list = []
    flag = False
    for _ in customer_account_list:
        num_transactions = random.randint(*customer_transaction_freq_range)
        for i in range(num_transactions):
            from_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            to_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            while from_account == to_account:
                to_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*customer_amount_range[get_customer_amount_range_bin()])
            fee_amount = random.uniform(*customer_fee_range)
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[0], sender_before_transaction_balance,
                                     'Customer', 'Customer', receiver_before_transaction_balance, anomaly])
        if flag:
            break

    return transaction_list


def generateCustomerToAgentTransaction(customer_account_list, agent_account_list, account_to_balance_map) -> list[list[str]]:
    global start_time
    global count
    print("Generating CW2AW transaction.....")
    transaction_list = []
    flag = False
    for _ in customer_account_list:
        num_transactions = random.randint(*customer_transaction_freq_range)
        for i in range(num_transactions):
            from_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            to_account = agent_account_list[random.randint(0, len(agent_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*customer_amount_range[get_customer_amount_range_bin()])
            fee_amount = random.uniform(*agent_fee_range)
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[1], sender_before_transaction_balance,
                                     'Customer', 'Agent', receiver_before_transaction_balance, anomaly])
        if flag:
            break
    return transaction_list


def generateAgentToCustomerTransaction(customer_account_list, agent_account_list, account_to_balance_map) -> list[
    list[str]]:
    global start_time
    global count
    print("Generating AW2CW transaction.....")
    flag = False
    transaction_list = []
    for _ in agent_account_list:
        num_transactions = random.randint(*agent_transaction_freq_range)
        for i in range(num_transactions):
            from_account = agent_account_list[random.randint(0, len(agent_account_list) - 1)]
            to_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*agent_amount_range[get_agent_amount_range_bin()])
            fee_amount = 0
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[2], sender_before_transaction_balance,
                                     'Agent', 'Customer', receiver_before_transaction_balance, anomaly])
        if flag:
            break
    return transaction_list


def generateCustomerToMerchantTransaction(customer_account_list, merchant_account_list, account_to_balance_map) -> list[
    list[str]]:
    global start_time
    global count
    print("Generating CW2MW transaction.....")
    flag = False
    transaction_list = []
    for _ in customer_account_list:
        num_transactions = random.randint(*merchant_transaction_freq_range)
        for i in range(num_transactions):
            from_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            to_account = merchant_account_list[random.randint(0, len(merchant_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*merchant_amount_range[get_merchant_amount_range_bin()])
            fee_amount = random.uniform(*merchant_fee_range)
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[3], sender_before_transaction_balance,
                                     'Customer', 'Merchant', receiver_before_transaction_balance, anomaly])
        if flag:
            break
    return transaction_list


def generateCustomerToBankTransaction(customer_account_list, bank_account_list, account_to_balance_map) -> list[
    list[str]]:
    global start_time
    global count
    print("Generating CW2CB transaction.....")
    transaction_list = []
    flag = False
    for _ in customer_account_list:
        num_transactions = random.randint(*customer_transaction_freq_range)
        for i in range(num_transactions):
            from_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            to_account = bank_account_list[random.randint(0, len(bank_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*customer_amount_range[get_customer_amount_range_bin()])
            fee_amount = random.uniform(*customer_fee_range)
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[4], sender_before_transaction_balance,
                                     'Customer', 'Bank', receiver_before_transaction_balance, anomaly])
        if flag:
            break

    return transaction_list


def generateBankToCustomerTransaction(customer_account_list, bank_account_list, account_to_balance_map) -> list[list[str]]:
    global start_time
    global count
    print("Generating CB2CW transaction....")
    transaction_list = []
    flag = False
    for _ in bank_account_list:
        num_transactions = random.randint(*customer_transaction_freq_range)
        for i in range(num_transactions):
            from_account = bank_account_list[random.randint(0, len(bank_account_list) - 1)]
            to_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*customer_amount_range[get_customer_amount_range_bin()])
            fee_amount = random.uniform(*customer_fee_range)
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[5], sender_before_transaction_balance,
                                     'Bank', 'Customer', receiver_before_transaction_balance, anomaly])
        if flag:
            break

    return transaction_list


def generateCardToCustomerTransaction(customer_account_list, card_number_list, account_to_balance_map) -> list[
    list[str]]:
    global start_time
    global count
    print("Generating CC2CB transaction .....")
    transaction_list = []
    flag = False
    for _ in card_number_list:
        num_transactions = random.randint(*customer_transaction_freq_range)
        for i in range(num_transactions):
            from_account = card_number_list[random.randint(0, len(card_number_list) - 1)]
            to_account = customer_account_list[random.randint(0, len(customer_account_list) - 1)]
            anomaly = is_generate_anomaly()
            anomaly_type = type_of_anomaly()
            if anomaly and anomaly_type == 0:
                amount = random.uniform(*customer_anomaly_amount_range)
            else:
                amount = random.uniform(*customer_amount_range[get_customer_amount_range_bin()])
            fee_amount = random.uniform(*customer_fee_range)
            sender_debit_amount = amount + fee_amount
            if account_to_balance_map[from_account] < sender_debit_amount:
                count += 1
                continue
            if anomaly and anomaly_type == 1:
                transaction_datetime = generate_anomaly_date_time(start_time)
            else:
                transaction_datetime = generate_date_time(start_time)
            start_time = transaction_datetime
            if transaction_datetime > today:
                flag = True
                break
            receiver_credit_amount = amount
            sender_before_transaction_balance = account_to_balance_map[from_account]
            receiver_before_transaction_balance = account_to_balance_map[to_account]
            account_to_balance_map[from_account] -= sender_debit_amount
            account_to_balance_map[to_account] += receiver_credit_amount
            depositor_running_balance = account_to_balance_map[from_account]
            withdrawer_running_balance = account_to_balance_map[to_account]
            transaction_list.append([transaction_datetime, amount, fee_amount, receiver_credit_amount,
                                     sender_debit_amount, depositor_running_balance, withdrawer_running_balance,
                                     from_account, to_account, transaction_type[6], sender_before_transaction_balance,
                                     'Card', 'Customer', receiver_before_transaction_balance, anomaly])
        if flag:
            break

    return transaction_list


def combineAllTransactionAndGenerateDataFrame(CW2CW, CW2AW, AW2CW, CW2MW, CW2CB, CB2CW, CC2CW, columns) -> DataFrame:
    print("Making dataframe from all transaction list.....")
    all_transaction = CW2AW + CW2MW + CW2CW + AW2CW + CW2CB + CB2CW + CC2CW

    all_transaction = sorted(all_transaction, key=lambda x: x[0])

    return pd.DataFrame(all_transaction, columns=columns)


def dataFrameToCsv(df) -> None:
    print("Converting dataframe to csv....")
    df.to_csv("transaction_report_2020_2024_with_anomaly_v2.csv", index=False)


if __name__ == "__main__":
    # Define transaction parameters
    customer_amount_range = [(0.1, 300), (300.01, 500), (501, 800), (800.01, 1000),
                             (1000.01, 1100)]  # [25%, 35%, 20%, 15%, 5%]
    # customer_amount_range = (0.1, 1000)
    customer_anomaly_amount_range = (1500, 5000)
    customer_transaction_freq_range = (0, 7)
    customer_transaction_time_range = (10, 20)  # 10 AM to 8 PM
    customer_initial_balance_range = (100000, 500000)
    customer_fee_range = (0.5, 10)

    agent_amount_range = [(10, 350), (350.01, 750), (750.01, 1000), (1001.01, 1250),
                          (1250.01, 1500)]  # [25%, 30%, 20%, 15%, 10%]
    # agent_amount_range = (10, 1500)
    agent_anomaly_amount_range = (2000, 5000)
    agent_transaction_freq_range = (0, 100)
    agent_transaction_time_range = (8, 22)  # 8 AM to 10 PM
    agent_initial_balance_range = (100000, 500000)
    agent_fee_range = (0.00, 9.5)

    merchant_amount_range = [(5, 400), (400.01, 750), (800.01, 1050), (1050.01, 1250), (1250.01, 1500),
                             (1500.01, 1600)]  # [20%, 25%, 20%, 20%, 10%, 5%]
    # merchant_amount_range = (5, 1600)
    merchant_anomaly_amount_range = (2000, 5000)
    merchant_transaction_freq_range = (0, 163)
    merchant_transaction_time_range = (9, 23)  # 9 AM to 11 PM
    merchant_balance_range = (0, 930000)
    merchant_initial_balance_range = (100000, 500000)
    merchant_fee_range = (0.00, 5.5)
    start_time = datetime.strptime("2018-01-01 09:00:00", "%Y-%m-%d %H:%M:%S")

    columns = ['transaction_date_time', 'amount', 'fee_amount', 'receiver_credit_amount',
               'sender_debit_amount', 'depositor_running_balance', 'withdrawer_running_balance',
               'from_account', 'to_account', 'transaction_type', 'sender_before_transaction_balance',
               'sender_account_type', 'receiver_account_type', 'receiver_before_transaction_balance', 'is_anomaly']
    transaction_type = ['CW2CW', 'CW2AW', 'AW2CW', 'CW2MW', 'CW2CB', 'CB2CW', 'CC2CW']
    dataFrame = pd.read_excel("transaction_report.xlsx")
    cA, aA, mA, bA, cN = getAllTypeOfAccountList(dataFrame)
    accountToBalanceMap = {}
    accountToBalanceMap = initializeAccountBalance(cA, customer_initial_balance_range, accountToBalanceMap)
    accountToBalanceMap = initializeAccountBalance(aA, agent_initial_balance_range, accountToBalanceMap)
    accountToBalanceMap = initializeAccountBalance(mA, merchant_initial_balance_range, accountToBalanceMap)
    accountToBalanceMap = initializeAccountBalance(bA, customer_initial_balance_range, accountToBalanceMap)
    accountToBalanceMap = initializeAccountBalance(cN, customer_initial_balance_range, accountToBalanceMap)

    CW2CW, CW2AW, AW2CW, CW2MW, CW2CB, CB2CW, CC2CW = [], [], [], [], [], [], []
    epoch = 500
    count = 0
    today = datetime.today()

    for i in range(epoch):
        print("Generating data epoc {}".format(i + 1))
        customer_to_customer_transaction_list = generateCustomerToCustomerTransaction(customer_account_list=cA,
                                                                                      account_to_balance_map=accountToBalanceMap)
        CW2CW.extend(customer_to_customer_transaction_list)
        customer_to_agent_transaction_list = generateCustomerToAgentTransaction(customer_account_list=cA,
                                                                                agent_account_list=aA,
                                                                                account_to_balance_map=accountToBalanceMap)
        CW2AW.extend(customer_to_agent_transaction_list)
        agent_to_customer_transaction_list = generateAgentToCustomerTransaction(customer_account_list=cA,
                                                                                agent_account_list=aA,
                                                                                account_to_balance_map=accountToBalanceMap)
        AW2CW.extend(agent_to_customer_transaction_list)
        customer_to_merchant_transaction_list = generateCustomerToMerchantTransaction(customer_account_list=cA,
                                                                                      merchant_account_list=mA,
                                                                                      account_to_balance_map=accountToBalanceMap)
        CW2MW.extend(customer_to_merchant_transaction_list)
        customer_to_bank_transaction_list = generateCustomerToBankTransaction(customer_account_list=cA,
                                                                              bank_account_list=bA,
                                                                              account_to_balance_map=accountToBalanceMap)
        CW2CB.extend(customer_to_bank_transaction_list)
        bank_to_customer_transaction_list = generateBankToCustomerTransaction(customer_account_list=cA,
                                                                              bank_account_list=bA,
                                                                              account_to_balance_map=accountToBalanceMap)
        CB2CW.extend(bank_to_customer_transaction_list)
        card_to_customer_transaction_list = generateCardToCustomerTransaction(customer_account_list=cA,
                                                                              card_number_list=cN,
                                                                              account_to_balance_map=accountToBalanceMap)
        CC2CW.extend(card_to_customer_transaction_list)

        print("_______________________________________")
        if start_time > today:
            break

    generated_dataFrame = combineAllTransactionAndGenerateDataFrame(CW2CW, CW2AW, AW2CW,
                                                                    CW2MW, CW2CB, CB2CW,
                                                                    CC2CW, columns)
    dataFrameToCsv(generated_dataFrame)

    print("Total cancel transaction: ", count)
